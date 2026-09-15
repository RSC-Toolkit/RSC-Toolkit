"""
AMS-D 학습 상태 콜백 — 모델(rsc_toolkit/models/ast_amsd.py) 의 epoch 인식과 per-sample 버퍼 크기.

- setup(stage=fit): datamodule 의 train 길이로 버퍼(all_predictions/ids_masked) 크기를 보정한다.
  Lightning 은 setup 훅 **뒤에** last.ckpt 를 strict 로 복원하므로 여기서 맞춰야 재개가 된다 —
  예전엔 on_fit_start(복원 뒤) 에서 보정해 SPRSound(6,656 ≠ 기본 4,142) 재개가 size mismatch 로 반드시 실패했다
  (REVIEW 2026-09-10 §4/§5 P1).
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
        # datamodule.setup("fit") 은 이 훅보다 먼저 불린다 (Lightning _call_setup_hook 순서).
        self._fit_buffers_to_datamodule(trainer, pl_module)

    def on_fit_start(self, trainer, pl_module):
        model = self._model(pl_module)
        # 모델이 "epoch 콜백이 붙어 있다"를 알 수 있게 표식을 남긴다 — 콜백 없이
        # model=ast_amsd 만 조합하면 epoch 이 영원히 0 이라 조용히 퇴화 학습이 되는데,
        # 모델 쪽 training_forward 가 이 표식의 부재를 보고 경고한다.
        model._epoch_callback_attached = True
        # setup 이 불리지 않은 경로(직접 fit 호출 없이 콜백만 붙인 테스트 등)의 안전망 — 재개 뒤엔 no-op.
        self._fit_buffers_to_datamodule(trainer, pl_module)
        logger.info("AMSD 버퍼: N=%d, masked=%d, epoch=%d", model.all_predictions.shape[0], model.ids_masked.shape[1], int(model.current_epoch))

    def on_train_epoch_start(self, trainer, pl_module):
        model = self._model(pl_module)
        model.current_epoch.copy_(torch.tensor(int(trainer.current_epoch), dtype=torch.long))
