from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List

import hydra
from omegaconf import DictConfig, OmegaConf

from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter
from rsc_toolkit.trainer.factories.build_model import build_model
from rsc_toolkit.trainer.factories.build_loss import build_loss
from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
from rsc_toolkit.trainer.factories.build_trainer import build_trainer


def _to_plain_dict(cfg: DictConfig) -> Dict[str, Any]:
    # Convert Hydra/OmegaConf objects into plain python dict/list types
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _get_loggers(trainer: L.Trainer) -> List[Any]:
    loggers: Iterable[Any] = getattr(trainer, "loggers", None) or []
    if not loggers:
        single = getattr(trainer, "logger", None)
        return [single] if single else []
    return list(loggers)


def _collect_run_metadata(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    run_cfg = cfg_dict.get("run", {})
    data_cfg = cfg_dict.get("data", {})
    data_params = data_cfg.get("params", {})
    features_cfg = data_params.get("features", {}) if isinstance(data_params, dict) else {}

    tags = run_cfg.get("tags") or []
    tag_str = ",".join(tags) if isinstance(tags, list) else tags

    model_name = cfg_dict.get("model", {}).get("name")
    data_name = data_cfg.get("name")
    run_name = run_cfg.get("name")
    if not run_name and model_name and data_name:
        # just some reasonable default if nothing provided.
        run_name = f"{data_name}-{model_name}"

    metadata = {
        "run/name": run_name,
        "run/group": run_cfg.get("group"),
        "run/tags": tag_str,
        "model/name": model_name,
        "data/name": data_name,
        "data/sample_level": data_params.get("sample_level"),
        "data/feature_kind": features_cfg.get("kind"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _log_run_metadata(trainer: L.Trainer, cfg_dict: Dict[str, Any]) -> None:
    metadata = _collect_run_metadata(cfg_dict)
    if not metadata:
        return
    for logger in _get_loggers(trainer):
        logger.log_hyperparams(metadata)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Optional: print fully resolved config for debugging
    print(OmegaConf.to_yaml(cfg, resolve=True))

    cfg_dict = _to_plain_dict(cfg)

    # --- Build components from config ---
    model = build_model(cfg_dict["model"])
    loss_func = build_loss(cfg_dict["loss"])

    train_metric_fns, val_metric_fns = build_train_val_metric_factories(cfg_dict.get("metrics", {}))

    datamodule = build_datamodule(cfg_dict["data"])
    trainer = build_trainer(cfg_dict["trainer"])

    # --- LightningModule ---
    pl_module = PLAdapter(
        rsc_model=model,
        loss_func=loss_func,
        train_metric_fns=train_metric_fns,
        val_metric_fns=val_metric_fns,
        optimizer_cfg=cfg_dict["optimizer"],  # see PLAdapter tweak below
    )

    _log_run_metadata(trainer, cfg_dict)

    # --- Run ---
    trainer.fit(pl_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
