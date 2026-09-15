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
    """이 run 이름으로 이미 저장된 산출물(best.pth/best.json/final_state.pth …) — 콜백의 dirpath/filename 에서 찾는다.

    같은 run.name 을 다시 투입하면 예전엔 아무 경고 없이 best.pth 를 덮어썼다 (REVIEW 2026-09-10 §4 P2).
    재개용 last.ckpt(EpochResumeCheckpoint)는 제외한다 — 그것은 이어가기 신호다.
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
        path = getattr(cb, "path", None)          # GapAugFinalMetrics 류: 파일 경로 하나
        if isinstance(path, (str, pathlib.Path)) and pathlib.Path(path).is_file():
            found.add(pathlib.Path(path))
    return sorted(found)


def guard_existing_run(trainer: L.Trainer, cfg_dict: Dict[str, Any], resuming: bool) -> None:
    """새로 시작하는 학습이 기존 산출물을 덮어쓰려 하면 멈춘다 (`+run.overwrite=true` 로 허용)."""
    if resuming:
        return
    existing = existing_run_artifacts(trainer)
    if not existing:
        return
    run_cfg = cfg_dict.get("run") or {}
    if bool(run_cfg.get("overwrite", False)):
        logger.warning("run.overwrite=true — 같은 run.name 의 기존 산출물을 덮어쓴다: %s", ", ".join(map(str, existing)))
        return
    raise RuntimeError(
        f"run.name={run_cfg.get('name')!r} 의 산출물이 이미 있다: {', '.join(map(str, existing))}. "
        "그대로 시작하면 best.pth/best.json 을 덮어쓴다 — 새 run.name 을 쓰거나, 이어가려면 last.ckpt 가 있는 상태에서 "
        "resume=auto(기본)로 투입하고, 정말 덮어쓰려면 +run.overwrite=true 를 붙여라."
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
    # adapter: 기본 pl. {name: split_iter, params: {split_iter: 4}} 면 분할 배치 어댑터 (M2D 미세조정 quirk).
    # 이름은 ADAPTERS registry 에서 찾고, 미지 이름은 침묵 폴백 없이 KeyError.
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
    # resume: "auto"(기본) 면 EpochResumeCheckpoint 의 last.ckpt 가 있을 때 이어간다.
    # "never" 면 무시하고 처음부터 (CLI: +resume=never).
    ckpt_path = None
    resume = str(cfg_dict.get("resume", "auto"))
    if resume == "auto":
        from rsc_toolkit.trainer.callbacks.resume_checkpoint import find_resume_path

        ckpt_path = find_resume_path(trainer)
        if ckpt_path is not None:
            logger.info("재개: %s 에서 이어서 학습한다", ckpt_path)
    elif resume != "never":
        raise ValueError(f"resume 은 auto|never 여야 한다: {resume!r}")

    guard_existing_run(trainer, cfg_dict, resuming=ckpt_path is not None)

    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)


if __name__ == "__main__":
    main()
