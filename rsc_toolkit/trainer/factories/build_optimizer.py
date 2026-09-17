# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Any, Dict

import torch.optim as optim

from rsc_toolkit.registries import OPTIMIZERS


def build_optimizer(cfg: Dict[str, Any], parameters) -> optim.Optimizer:
    """
    cfg example:
      {"name": "adam", "params": {"lr": 1e-3, "weight_decay": 0.0}}

    lr_scheduler is handled by PLAdapter.configure_optimizers (it must not appear in this cfg).
    An unknown name raises KeyError listing the available names. params is copied before use, so cfg is not modified.
    """
    return OPTIMIZERS.build(cfg, parameters)
