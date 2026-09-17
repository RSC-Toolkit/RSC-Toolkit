# Portions of this file are adapted from AMS-D (https://github.com/CcnNnn/AMS-D); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
AMS-D training-state callback: epoch awareness and per-sample buffer sizes of the model (rsc_toolkit/models/ast_amsd.py).

- setup(stage=fit): resizes the buffers (all_predictions/ids_masked) to the datamodule's train length.
  Lightning restores last.ckpt strictly **after** the setup hook, so the sizes must be fixed here for resume to work;
  resizing later (e.g. in on_fit_start) makes a SPRSound resume (6,656 ≠ default 4,142) fail with a size mismatch.
- on_train_epoch_start: model.current_epoch ← trainer.current_epoch (the epoch argument of the original main.py:210-226; 0 means no masking and a one-hot teacher).
"""
from __future__ import annotations

import logging

import lightning as L
import torch

logger = logging.getLogger(__name__)


class AMSDEpochCallback(L.Callback):
    def __init__(self, model_attr: str = "rsc_model"):
        super().__init__()
        self.model_attr = model_attr

    def _model(self, pl_module):
        return getattr(pl_module, self.model_attr, pl_module)

    def _fit_buffers_to_datamodule(self, trainer, pl_module) -> None:
        model = self._model(pl_module)
        dm = trainer.datamodule
        n = getattr(dm, "num_train_samples", None)
        if n is None and getattr(dm, "train_ds", None) is not None:
            n = len(dm.train_ds)
        if n is not None and hasattr(model, "ensure_buffers"):
            model.ensure_buffers(int(n))

    def setup(self, trainer, pl_module, stage=None):
        if stage not in (None, "fit"):
            return
        # datamodule.setup("fit") is called before this hook (Lightning _call_setup_hook order).
        self._fit_buffers_to_datamodule(trainer, pl_module)

    def on_fit_start(self, trainer, pl_module):
        model = self._model(pl_module)
        # Leave a marker so the model knows the epoch callback is attached. Using
        # model=ast_amsd without this callback keeps epoch at 0 forever and silently degrades training;
        # the model's training_forward warns when this marker is missing.
        model._epoch_callback_attached = True
        # Safety net for paths where setup is not called (e.g. tests that attach the callback without fit); no-op after resume.
        self._fit_buffers_to_datamodule(trainer, pl_module)
        logger.info("AMSD buffers: N=%d, masked=%d, epoch=%d", model.all_predictions.shape[0], model.ids_masked.shape[1], int(model.current_epoch))

    def on_train_epoch_start(self, trainer, pl_module):
        model = self._model(pl_module)
        model.current_epoch.copy_(torch.tensor(int(trainer.current_epoch), dtype=torch.long))
