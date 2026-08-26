"""
QLung 의 MA(--ma_update --ma_beta 0.6) — LiveWeightAveraging 에 **제외 접두어**를 더한 서브클래스.

train_qlung.py:383-393,490-499: 매 iteration 직전 state_dict 를 스냅샷해 optimizer.step 뒤
  new = 0.6·prev + 0.4·cur 로 model·classifier·projector 만 덮어쓴다. criterion[1] = DFAM(AngularMarginLoss) 의 W_d 는
  optimizer 에는 들어가지만(train_qlung.py:345) MA 는 받지 않는다. 툴킷은 DFAM 을 모델 서브모듈 `dfam` 으로 두므로
  여기서 `dfam.` 접두어 키를 평균에서 제외한다(현재 값 유지). 제외 목록은 인자 또는 모델의 ma_exclude_prefixes 속성.
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
