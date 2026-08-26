"""
AMS-D 학습 상태 콜백 — 모델(rsc_toolkit/models/ast_amsd.py) 의 epoch 인식과 per-sample 버퍼 크기.

- on_fit_start: datamodule 의 train 길이로 버퍼(all_predictions/ids_masked) 크기를 보정한다(재개 시 ckpt 로드가 먼저라 no-op).
- on_train_epoch_start: model.current_epoch ← trainer.current_epoch (원본 main.py:210-226 의 epoch 인자; 0 이면 마스킹 없음·one-hot teacher).
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

    def on_fit_start(self, trainer, pl_module):
        model = self._model(pl_module)
        dm = trainer.datamodule
        n = getattr(dm, "num_train_samples", None)
        if n is None and getattr(dm, "train_ds", None) is not None:
            n = len(dm.train_ds)
        if n is not None and hasattr(model, "ensure_buffers"):
            model.ensure_buffers(int(n))
        logger.info("AMSD 버퍼: N=%d, masked=%d, epoch=%d", model.all_predictions.shape[0], model.ids_masked.shape[1], int(model.current_epoch))

    def on_train_epoch_start(self, trainer, pl_module):
        model = self._model(pl_module)
        model.current_epoch.copy_(torch.tensor(int(trainer.current_epoch), dtype=torch.long))
