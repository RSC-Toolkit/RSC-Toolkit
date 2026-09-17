"""
QLung MA (--ma_update --ma_beta 0.6): a LiveWeightAveraging subclass with **excluded prefixes**.

train_qlung.py:383-393,490-499: snapshots the state_dict before every iteration and, after optimizer.step, overwrites
  only model·classifier·projector with new = 0.6·prev + 0.4·cur. W_d of criterion[1] = DFAM(AngularMarginLoss) is
  in the optimizer (train_qlung.py:345) but does not receive MA. The toolkit holds DFAM as the model submodule `dfam`, so
  keys with the `dfam.` prefix are excluded from averaging here (current values kept). The exclusion list is the argument or the model's ma_exclude_prefixes attribute.
"""
from typing import Optional, Sequence

import torch

from .ma_update import LiveWeightAveraging


class LiveWeightAveragingExcluding(LiveWeightAveraging):
    def __init__(self, beta: float = 0.6, model_attr: str = "rsc_model",
                 exclude_prefixes: Optional[Sequence[str]] = None):
        super().__init__(beta=beta, model_attr=model_attr)
        self.exclude_prefixes = tuple(exclude_prefixes) if exclude_prefixes is not None else None

    def _excluded(self, model) -> tuple:
        if self.exclude_prefixes is not None:
            return self.exclude_prefixes
        return tuple(getattr(model, "ma_exclude_prefixes", ()) or ())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._prev is None:
            return
        model = self._target(pl_module)
        excluded = self._excluded(model)
        with torch.no_grad():
            current = model.state_dict()
            blended = {}
            for k, v in current.items():
                if excluded and k.startswith(excluded):
                    blended[k] = v
                else:
                    blended[k] = self._prev[k] * self.beta + (1 - self.beta) * v
            model.load_state_dict(blended)
        self._prev = None
