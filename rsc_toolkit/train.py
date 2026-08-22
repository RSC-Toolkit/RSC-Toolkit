from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter
from rsc_toolkit.trainer.factories.build_model import build_model
from rsc_toolkit.trainer.factories.build_loss import build_loss
from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
from rsc_toolkit.trainer.factories.build_trainer import build_trainer


logger = logging.getLogger(__name__)


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


def _apply_seed(cfg_dict: Dict[str, Any]) -> None:
    """
    config 의 seed 를 실제로 적용한다.

    이전에는 config 에 `seed: 123` 이 있어도 코드에서 아무도 읽지 않아
    매 실행이 다른 난수로 돌았다. 모델 초기화보다 먼저 불러야 한다.
    workers=True 로 DataLoader 워커의 시드까지 파생시킨다.
    """
    seed = cfg_dict.get("seed")
    if seed is None:
        logger.warning("config 에 seed 가 없다. 실행마다 결과가 달라진다.")
        return
    L.seed_everything(int(seed), workers=True)
    logger.info("seed=%s 적용 (workers=True)", seed)


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

    # 난수를 쓰는 첫 단계(모델 초기화)보다 반드시 앞에 와야 한다.
    _apply_seed(cfg_dict)

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
