"""
매 iteration 직전 가중치와의 가중 평균 — patch-mix 계열 논문의 --ma_update 재현.

출처: raymin0223/patch-mix util/misc.py update_moving_average + main.py train().
원본은 batch 시작 때 state_dict 를 통째로 복사해 두고, optimizer.step() 뒤
  new = old * beta + (1 - beta) * current
로 **live 모델을 직접** 덮어쓴다 (shadow EMA 가 아니다). buffers(BN 통계 등)도
state_dict 에 포함되므로 같이 평균된다 — 원본과 동일.

툴킷 기본 EMACallback(beta 0.999, shadow) 과는 별개다. 논문 세팅(beta 0.5)은
이 콜백을 쓴다.
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
