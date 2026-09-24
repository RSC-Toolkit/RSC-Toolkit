# Portions of this file are adapted from patch-mix_contrastive_learning (https://github.com/raymin0223/patch-mix_contrastive_learning); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Weighted average with the weights from just before each iteration: --ma_update of the patch-mix family papers.

Source: raymin0223/patch-mix util/misc.py update_moving_average + main.py train().
The original code copies the whole state_dict at batch start and, after optimizer.step(),
overwrites **the live model directly** with
  new = old * beta + (1 - beta) * current
(not a shadow EMA). Buffers (BN statistics, etc.) are part of the
state_dict and are averaged too, same as the original code.

Separate from the toolkit's default EMACallback (beta 0.999, shadow). The paper setting (beta 0.5)
uses this callback.
"""
from typing import Dict, Optional

import lightning as L
import torch


class LiveWeightAveraging(L.Callback):
    def __init__(self, beta: float = 0.5, model_attr: str = "rsc_model"):
        super().__init__()
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = beta
        self.model_attr = model_attr
        self._prev: Optional[Dict[str, torch.Tensor]] = None

    def _target(self, pl_module):
        return getattr(pl_module, self.model_attr, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        model = self._target(pl_module)
        with torch.no_grad():
            self._prev = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._prev is None:
            return
        model = self._target(pl_module)
        with torch.no_grad():
            current = model.state_dict()
            blended = {
                k: self._prev[k] * self.beta + (1 - self.beta) * v for k, v in current.items()
            }
            model.load_state_dict(blended)
        self._prev = None
