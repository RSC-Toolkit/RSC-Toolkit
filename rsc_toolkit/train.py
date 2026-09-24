# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Iterable, List

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from rsc_toolkit.trainer.factories.build_adapter import build_adapter
from rsc_toolkit.trainer.factories.build_model import build_model
from rsc_toolkit.trainer.factories.build_loss import build_loss
from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
from rsc_toolkit.trainer.factories.build_trainer import build_trainer


logger = logging.getLogger(__name__)


def existing_run_artifacts(trainer: L.Trainer) -> List[pathlib.Path]:
    """Artifacts already saved under this run name (best.pth/best.json/final_state.pth …), found via the callbacks' dirpath/filename.

    Used to prevent a run with the same run.name from silently overwriting best.pth.
    The resume last.ckpt (EpochResumeCheckpoint) is excluded — it signals that training should continue.
    """
    from rsc_toolkit.trainer.callbacks.resume_checkpoint import EpochResumeCheckpoint

    found = set()
    for cb in trainer.callbacks:
        if isinstance(cb, EpochResumeCheckpoint):
            continue
        dirpath, filename = getattr(cb, "dirpath", None), getattr(cb, "filename", None)
        if dirpath is not None and filename is not None:
            d = pathlib.Path(dirpath)
            for name in {str(filename), "best.json", pathlib.Path(str(filename)).with_suffix(".json").name}:
                if (d / name).is_file():
                    found.add(d / name)
        path = getattr(cb, "path", None)          # GapAugFinalMetrics and similar: a single file path
        if isinstance(path, (str, pathlib.Path)) and pathlib.Path(path).is_file():
            found.add(pathlib.Path(path))
    return sorted(found)


def guard_existing_run(trainer: L.Trainer, cfg_dict: Dict[str, Any], resuming: bool) -> None:
    """Stop if a freshly started run would overwrite existing artifacts (allowed with `+run.overwrite=true`)."""
    if resuming:
        return
    existing = existing_run_artifacts(trainer)
    if not existing:
        return
    run_cfg = cfg_dict.get("run") or {}
    if bool(run_cfg.get("overwrite", False)):
        logger.warning("run.overwrite=true — overwriting existing artifacts of the same run.name: %s", ", ".join(map(str, existing)))
        return
    raise RuntimeError(
        f"artifacts for run.name={run_cfg.get('name')!r} already exist: {', '.join(map(str, existing))}. "
        "Starting now would overwrite best.pth/best.json — use a new run.name; to continue, launch with "
        "resume=auto (default) while last.ckpt exists; to really overwrite, add +run.overwrite=true."
    )


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
    Apply the seed from the config.

    Must be called before model initialization.
    workers=True also derives the seeds of DataLoader workers.
    """
    seed = cfg_dict.get("seed")
    if seed is None:
        logger.warning("no seed in config; results will differ between runs.")
        return
    L.seed_everything(int(seed), workers=True)
    logger.info("seed=%s applied (workers=True)", seed)


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

    # Must come before the first step that uses random numbers (model initialization).
    _apply_seed(cfg_dict)

    # --- Build components from config ---
    model = build_model(cfg_dict["model"])
    loss_func = build_loss(cfg_dict["loss"])

    train_metric_fns, val_metric_fns = build_train_val_metric_factories(cfg_dict.get("metrics", {}))

    datamodule = build_datamodule(cfg_dict["data"])
    trainer = build_trainer(cfg_dict["trainer"])

    # --- LightningModule ---
    # adapter: default pl. {name: split_iter, params: {split_iter: 4}} selects the split-batch adapter (M2D fine-tuning quirk).
    # Names are looked up in the ADAPTERS registry; unknown names raise KeyError without a silent fallback.
    pl_module = build_adapter(
        cfg_dict.get("adapter"),
        rsc_model=model,
        loss_func=loss_func,
        train_metric_fns=train_metric_fns,
        val_metric_fns=val_metric_fns,
        optimizer_cfg=cfg_dict["optimizer"],
    )

    _log_run_metadata(trainer, cfg_dict)

    # --- Run ---
    # resume: "auto" (default) continues from EpochResumeCheckpoint's last.ckpt if it exists.
    # "never" ignores it and starts from scratch (CLI: +resume=never).
    ckpt_path = None
    resume = str(cfg_dict.get("resume", "auto"))
    if resume == "auto":
        from rsc_toolkit.trainer.callbacks.resume_checkpoint import find_resume_path

        ckpt_path = find_resume_path(trainer)
        if ckpt_path is not None:
            logger.info("resume: continuing training from %s", ckpt_path)
    elif resume != "never":
        raise ValueError(f"resume must be auto|never: {resume!r}")

    guard_existing_run(trainer, cfg_dict, resuming=ckpt_path is not None)

    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)


if __name__ == "__main__":
    main()
