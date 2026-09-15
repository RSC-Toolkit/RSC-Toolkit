"""
RespireNet 평가·모델 선택 프로토콜 — 원본 train.py evaluate()/체크포인트 규칙 재현.

매 epoch 끝에 test 로더 전체를 다시 forward 해 청크 예측을 identifier(cycle) 별로 다수결
(np.bincount argmax — 동률이면 낮은 클래스) 집계하고, cycle 단위 accuracy / Sp / Se / Score 를 로그한다.
best 저장 기준은 **cycle accuracy 최고** (train.py:266 `if best_acc < acc`) — 다른 묶음의 Score·Se>5 와 다르다.
Score 기준 best 도 참고로 함께 기록한다. 재개(EpochResumeCheckpoint)를 위해 상태를 state_dict 로 보존.
"""
from __future__ import annotations

import logging
import pathlib

import lightning as L

from rsc_toolkit.trainer.callbacks.atomic_io import atomic_json_dump, atomic_torch_save
import numpy as np
import torch

logger = logging.getLogger(__name__)


def vote_cycles(chunk_preds, identifiers, cycle_ids):
    preds, ids = np.asarray(chunk_preds), np.asarray(identifiers)
    order = {}
    for i, cid in enumerate(ids):
        order.setdefault(cid, []).append(i)
    y_true, y_pred = [], []
    for cid in cycle_ids:
        y_pred.append(int(np.argmax(np.bincount(preds[order[cid]]))))
        y_true.append(int(cid.split("_")[-1]))
    n_cls = max(max(y_true), max(y_pred)) + 1
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    sp = conf[0, 0] / max(conf[0].sum(), 1) * 100
    se = np.trace(conf[1:, 1:]) / max(conf[1:].sum(), 1) * 100
    acc = float(np.trace(conf) / conf.sum() * 100)
    return {"conf": conf.tolist(), "S_p": float(sp), "S_e": float(se), "icbhi_score": float((sp + se) / 2), "cycle_acc": acc}


class RespireNetVoteEval(L.Callback):
    def __init__(self, dirpath: str, filename: str = "best.pth", model_attr: str = "rsc_model"):
        super().__init__()
        self.dirpath = pathlib.Path(dirpath)
        self.filename = filename
        self.model_attr = model_attr
        self.best_acc = float("-inf")
        self.best_info = None
        self.best_score_info = None  # 참고용 (Score 기준)

    def state_dict(self):
        return {"best_acc": self.best_acc, "best_info": self.best_info, "best_score_info": self.best_score_info}

    def load_state_dict(self, state_dict):
        self.best_acc = float(state_dict.get("best_acc", float("-inf")))
        self.best_info = state_dict.get("best_info")
        self.best_score_info = state_dict.get("best_score_info")

    @torch.no_grad()
    def _evaluate(self, trainer, pl_module):
        ds = trainer.datamodule.test_ds
        loader = trainer.datamodule.val_dataloader()
        model = getattr(pl_module, self.model_attr, pl_module)
        was_training = model.training
        model.eval()
        preds = []
        for batch in loader:
            x = batch[0].to(pl_module.device)
            preds.extend(model(x).argmax(1).cpu().tolist())
        model.train(was_training)
        return vote_cycles(preds, ds.identifiers, ds.cycle_ids)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        res = self._evaluate(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        pl_module.log_dict({"vote/cycle_acc": res["cycle_acc"], "vote/S_p": res["S_p"], "vote/S_e": res["S_e"],
                            "vote/icbhi_score": res["icbhi_score"]}, prog_bar=True, on_epoch=True)
        info = {"epoch": epoch, **{k: v for k, v in res.items()}}
        if self.best_score_info is None or res["icbhi_score"] > self.best_score_info["icbhi_score"]:
            self.best_score_info = info
        if res["cycle_acc"] > self.best_acc:  # 원본: if best_acc < acc
            self.best_acc = res["cycle_acc"]
            self.best_info = info
            self.dirpath.mkdir(parents=True, exist_ok=True)
            model = getattr(pl_module, self.model_attr, pl_module)
            atomic_torch_save({"model": model.state_dict(), **info}, self.dirpath / self.filename)
            atomic_json_dump({**info, "best_score_info": self.best_score_info}, self.dirpath / "best.json")
            logger.info("Best(cycle acc) 갱신: epoch %d acc %.2f | Sp %.2f Se %.2f Score %.2f",
                        epoch, res["cycle_acc"], res["S_p"], res["S_e"], res["icbhi_score"])
        elif self.best_info is not None:
            # Score-best 참고 기록만 갱신
            atomic_json_dump({**self.best_info, "best_score_info": self.best_score_info}, self.dirpath / "best.json")
