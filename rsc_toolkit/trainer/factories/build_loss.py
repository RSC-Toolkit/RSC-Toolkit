from typing import Any, Dict

import torch.nn as nn

from rsc_toolkit.trainer.registries.loss_registry import LOSSES


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
      {"name": "cross_entropy", "params": {"label_smoothing": 0.0}}
    """
    name = cfg["name"]
    params = cfg.get("params", {})

    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(LOSSES.keys())}")

    cls = LOSSES[name]
    return cls(**params)
