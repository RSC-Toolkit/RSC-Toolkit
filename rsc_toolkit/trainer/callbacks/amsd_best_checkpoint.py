"""
AMS-D 용 best 선택 — ICBHIBestScoreCheckpoint 와 같은 프로토콜(val=공식 test, Score 최고 && Se > min_se) 이되
(1) 판정 키를 고를 수 있고(기본: 논문 이진 지표 paper_score/paper_S_e/paper_S_p — 사용자 결정 2026-08-26),
(2) best.json 에 표준 4-class ICBHIScore(S_p/S_e/icbhi_score) 와 이진 지표를 **모두** 기록한다.

표준 산식으로 best 를 고르려면 score_key=icbhi_score se_key=S_e sp_key=S_p 로 바꾼다.
공식 코드의 선택(train 내부 80/20 valid 의 best_acc/best_score ckpt, test.py 기본 epoch 45) 은 구현하지 않고 문서화한다.
"""
from __future__ import annotations

import json
import logging
from typing import Sequence

import torch

from .icbhi_best_checkpoint import ICBHIBestScoreCheckpoint

logger = logging.getLogger(__name__)

DEFAULT_RECORD_KEYS = ("S_p", "S_e", "icbhi_score", "paper_S_p", "paper_S_e", "paper_score", "paper_acc")


class AMSDBestScoreCheckpoint(ICBHIBestScoreCheckpoint):
    def __init__(
        self,
        dirpath: str = "outputs/amsd",
        filename: str = "best.pth",
        score_key: str = "paper_score",
        se_key: str = "paper_S_e",
        sp_key: str = "paper_S_p",
        min_se: float = -1.0,             # 원본에 Se 게이트 없음
        model_attr: str = "rsc_model",
        record_keys: Sequence[str] = DEFAULT_RECORD_KEYS,
    ):
        super().__init__(dirpath=dirpath, filename=filename, score_key=score_key, se_key=se_key, sp_key=sp_key,
                         min_se=min_se, model_attr=model_attr)
        self.record_keys = tuple(record_keys)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.score_key not in metrics:
            logger.warning("메트릭 %s 가 없어 best 선정을 건너뛴다 (metrics=%s)", self.score_key, list(metrics))
            return
        score = float(metrics[self.score_key])
        se = float(metrics.get(self.se_key, float("nan")))
        if not (score > self.best_score and se > self.min_se):
            return
        self.best_score = score
        info = {"epoch": trainer.current_epoch + 1, "selection_key": self.score_key}
        for k in self.record_keys:
            if k in metrics:
                info[k] = float(metrics[k])
        self.best_info = info
        self.dirpath.mkdir(parents=True, exist_ok=True)
        model = getattr(pl_module, self.model_attr, pl_module)
        path = self.dirpath / self.filename
        torch.save({"model": model.state_dict(), **info}, path)
        json.dump(info, open(path.with_suffix(".json"), "w"), indent=1)
        logger.info("Best 갱신(%s): epoch %d, %s=%.2f | 표준 Sp/Se/Score %s/%s/%s", self.score_key, info["epoch"], self.score_key, score,
                    info.get("S_p"), info.get("S_e"), info.get("icbhi_score"))
