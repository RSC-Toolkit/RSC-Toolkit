# Portions of this file are adapted from BTS-CARD (https://github.com/RSC-Toolkit/BTS-CARD); that code is under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
BTS-CARD callbacks.

- BTSCARDWeightAveraging: reproduces the per-step MA (β=0.5) of the original main:325-331, which averages only
  model/classifier/projector/meta_classifier and excludes the discriminators (loc/dev) and LearnableConstant.
  Subclasses LiveWeightAveraging (patch-mix family live MA) and filters state_dict keys by prefix.
- BTSCARDLossLogger: logs the per-term losses (o1/o2/kl/nde/loc/dev) left by the model's training_forward
  (corresponds to the AverageMeter output of the original train()).
"""
from typing import Optional, Sequence

import torch

from rsc_toolkit.trainer.callbacks.ma_update import LiveWeightAveraging
from rsc_toolkit.models.bts_card import MA_PREFIXES

import lightning as L


class BTSCARDWeightAveraging(LiveWeightAveraging):
    def __init__(self, beta: float = 0.5, model_attr: str = "rsc_model",
                 include_prefixes: Sequence[str] = MA_PREFIXES):
        super().__init__(beta=beta, model_attr=model_attr)
        self.include_prefixes = tuple(include_prefixes)

    def _is_target(self, key: str) -> bool:
        return key.startswith(self.include_prefixes)

    def target_keys(self, pl_module):
        return [k for k in self._target(pl_module).state_dict() if self._is_target(k)]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        model = self._target(pl_module)
        with torch.no_grad():
            self._prev = {k: v.detach().clone() for k, v in model.state_dict().items() if self._is_target(k)}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._prev is None:
            return
        model = self._target(pl_module)
        with torch.no_grad():
            current = model.state_dict()
            blended = {k: self._prev[k] * self.beta + (1 - self.beta) * current[k] for k in self._prev}
            model.load_state_dict(blended, strict=False)  # discriminator and constant keys are left unchanged
        self._prev = None


class BTSCARDLossLogger(L.Callback):
    def __init__(self, model_attr: str = "rsc_model", prefix: str = "train_loss_"):
        super().__init__()
        self.model_attr = model_attr
        self.prefix = prefix

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = getattr(pl_module, self.model_attr, pl_module)
        terms: Optional[dict] = getattr(model, "loss_terms", None)
        if terms:
            pl_module.log_dict({f"{self.prefix}{k}": float(v) for k, v in terms.items()},
                               on_step=True, on_epoch=True, prog_bar=False)
