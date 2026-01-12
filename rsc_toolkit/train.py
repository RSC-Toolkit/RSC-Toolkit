from __future__ import annotations

import os
from typing import Any, Dict

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


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
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

    # --- Run ---
    trainer.fit(pl_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
