from typing import Any, Dict

import torch.optim as optim

from rsc_toolkit.trainer.registries.optimizer_registry import OPTIMIZERS


def build_optimizer(cfg: Dict[str, Any], parameters) -> optim.Optimizer:
    """
    cfg example:
      {"name": "adam", "params": {"lr": 1e-3, "weight_decay": 0.0}}
    """
    name = cfg["name"]
    params = cfg.get("params", {})

    if name not in OPTIMIZERS:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {sorted(OPTIMIZERS.keys())}")

    cls = OPTIMIZERS[name]
    return cls(parameters, **params)
