"""
GaP-Aug 재현용: 학습 종료 시점(최종 iterate) 의 test 지표를 best.json 에 final_* 로 병기한다.

논문은 20,000 iterations 뒤 어떤 모델로 평가했는지 미기재(AUTHOR_QUESTIONS GaP-Aug #8) → 툴킷 기본 best-on-test(ICBHIBestScoreCheckpoint,
Se 게이트 없음) 와 함께 마지막 검증의 지표도 남겨 두 프로토콜을 모두 비교할 수 있게 한다.
acc@1(Table 1 Acc 69.01 보조 앵커) 도 final_acc@1 로 남긴다(best.json 의 best 항목에는 acc 가 없음).
콜백 목록에서 ICBHIBestScoreCheckpoint 뒤에 두어야 한다 (on_fit_end 시점에 best.json 이 있어야 병합).
"""
from __future__ import annotations

import json
import logging
import pathlib

import lightning as L

from rsc_toolkit.trainer.callbacks.atomic_io import atomic_json_dump

logger = logging.getLogger(__name__)


class FinalMetricsToBestJson(L.Callback):
    def __init__(self, dirpath: str, filename: str = "best.json", keys=("icbhi_score", "S_e", "S_p", "acc@1", "val_loss")):
        super().__init__()
        self.path = pathlib.Path(dirpath) / filename
        self.keys = tuple(keys)
        self._last = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self._last = {f"final_{k}": float(m[k]) for k in self.keys if k in m}
        self._last["final_epoch"] = trainer.current_epoch + 1
        self._last["final_global_step"] = int(trainer.global_step)

    def on_fit_end(self, trainer, pl_module):
        if not self._last:
            return
        data = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except Exception:  # pragma: no cover
                data = {}
        data.update(self._last)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(data, self.path)
        logger.info("최종 iterate 지표 기록 → %s: %s", self.path, self._last)
