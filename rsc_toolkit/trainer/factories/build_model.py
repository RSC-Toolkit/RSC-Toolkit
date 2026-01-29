from typing import Any, Dict

import torch.nn as nn

from rsc_toolkit.trainer.registries.model_registry import MODELS


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
      {"name": "audioclap", "params": {"n_cls": 4}}
    """
    name = cfg["name"]
    params = cfg.get("params", {})

    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS.keys())}")

    cls = MODELS[name]
    return cls(**params)
