# Portions of this file are adapted from patch-mix_contrastive_learning (https://github.com/raymin0223/patch-mix_contrastive_learning); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class MA:
    def __init__(self, beta: float):
        super().__init__()
        self.beta = float(beta)

    def update_average(self, old: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(
    ma_beta: float,
    current_model: nn.Module,
    ma_ckpt: Dict[str, Any],
) -> nn.Module:
    """
    Legacy helper used in older versions: updates `current_model` weights by blending
    in weights from `ma_ckpt` (EMA/MA checkpoint), then loads the blended weights.
    """
    ma_updater = MA(ma_beta)
    new_state_dict = {}
    for (k1, current_params), (k2, ma_params) in zip(
        current_model.state_dict().items(),
        ma_ckpt.items(),
    ):
        assert k1 == k2
        old_weight, up_weight = ma_params.data, current_params.data
        new_state_dict[k1] = ma_updater.update_average(old_weight, up_weight)

    current_model.load_state_dict(new_state_dict)
    return current_model

