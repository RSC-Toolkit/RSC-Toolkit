from typing import Any, Dict

import lightning as L

from rsc_toolkit.trainer.registries.data_registry import DATAMODULES


def build_datamodule(cfg: Dict[str, Any]) -> L.LightningDataModule:
    """
    cfg example:
      {"name": "icbhi", "params": {"batch_size": 32, "num_workers": 4, "fold": 1}}
    """
    name = cfg["name"]
    params = cfg.get("params", {})

    datamodule_name = params.pop('datamodule_name')

    if datamodule_name not in DATAMODULES:
        raise KeyError(f"Unknown datamodule '{name}'. Available: {sorted(DATAMODULES.keys())}")

    cls = DATAMODULES[datamodule_name]
    return cls(**params)
