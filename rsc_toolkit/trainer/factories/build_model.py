from typing import Any, Dict

import torch.nn as nn

from rsc_toolkit.registries import MODELS


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
      {"name": "audioclap", "params": {"n_cls": 4}}

    An unknown name raises KeyError listing the available names. params is copied before use, so cfg is not modified.
    """
    return MODELS.build(cfg)
