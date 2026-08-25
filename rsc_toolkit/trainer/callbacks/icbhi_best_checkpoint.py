"""
ICBHI 논문 프로토콜의 모델 선택 — patch-mix main.py validate()/저장 로직 재현.

원본은 검증셋 없이 매 epoch 공식 test 를 평가해 "Score 최고 && Se > 5" 일 때
model+classifier state 를 저장한다 (보고 수치는 사실상 best-on-test).
논문 수치와 비교하려면 같은 프로토콜이어야 하므로 그대로 재현한다.
val 데이터로더에 공식 test split 을 물리는 것(config 의 val_splits: [test])이 전제다.

ICBHIScore 메트릭이 로그하는 "icbhi_score" / "S_e" / "S_p" 를 읽는다.
"""
import json
import logging
import pathlib

import lightning as L
import torch

logger = logging.getLogger(__name__)


class ICBHIBestScoreCheckpoint(L.Callback):
    def __init__(
        self,
        dirpath: str = "outputs/checkpoints",
        filename: str = "best.pth",
        score_key: str = "icbhi_score",
        se_key: str = "S_e",
        sp_key: str = "S_p",
        min_se: float = 5.0,
        model_attr: str = "rsc_model",
    ):
        super().__init__()
        self.dirpath = pathlib.Path(dirpath)
        self.filename = filename
        self.score_key = score_key
        self.se_key = se_key
        self.sp_key = sp_key
        self.min_se = float(min_se)
        self.model_attr = model_attr
        self.best_score = float("-inf")
        self.best_info = None

    # 재개(EpochResumeCheckpoint) 시 best 기준이 이어져야 한다 — 없으면 재개 후 첫
    # epoch 가 무조건 best 로 덮어써 프로토콜(전 구간 Score 최고)이 깨진다.
    def state_dict(self):
        return {"best_score": self.best_score, "best_info": self.best_info}

    def load_state_dict(self, state_dict):
        self.best_score = float(state_dict.get("best_score", float("-inf")))
        self.best_info = state_dict.get("best_info")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.score_key not in metrics:
            logger.warning("메트릭 %s 가 없어 best 선정을 건너뛴다 (metrics=%s)", self.score_key, list(metrics))
            return
        score = float(metrics[self.score_key])
        se = float(metrics.get(self.se_key, float("nan")))
        sp = float(metrics.get(self.sp_key, float("nan")))

        # 원본: if sc > best_acc[-1] and se > 5
        if not (score > self.best_score and se > self.min_se):
            return

        self.best_score = score
        self.best_info = {"epoch": trainer.current_epoch + 1, "S_p": sp, "S_e": se, "icbhi_score": score}

        self.dirpath.mkdir(parents=True, exist_ok=True)
        model = getattr(pl_module, self.model_attr, pl_module)
        path = self.dirpath / self.filename
        torch.save({"model": model.state_dict(), **self.best_info}, path)
        json.dump(self.best_info, open(path.with_suffix(".json"), "w"), indent=1)
        logger.info("Best 갱신: epoch %d, Score %.2f (Sp %.2f / Se %.2f) -> %s",
                    self.best_info["epoch"], score, sp, se, path)
