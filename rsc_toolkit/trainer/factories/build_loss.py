# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Any, Dict

import torch.nn as nn

from rsc_toolkit.registries import LOSSES


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
      {"name": "cross_entropy", "params": {"label_smoothing": 0.0}}

    An unknown name raises KeyError listing the available names. params is copied before use, so cfg is not modified.
    """
    return LOSSES.build(cfg)
