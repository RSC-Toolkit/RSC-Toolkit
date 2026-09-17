# Portions of this file are adapted from RespireNet (https://github.com/microsoft/RespireNet); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
RespireNet evaluation and model-selection protocol, reproducing evaluate() and the checkpoint rule of the original train.py.

At the end of every epoch the whole test loader is forwarded again, chunk predictions are aggregated per identifier (cycle) by majority
vote (np.bincount argmax; ties go to the lower class), and cycle-level accuracy / Sp / Se / Score are logged.
The best checkpoint is selected by **highest cycle accuracy** (train.py:266 `if best_acc < acc`), unlike the Score·Se>5 rule of other groups.
The Score-based best is also recorded for reference. State is kept in state_dict for resume (EpochResumeCheckpoint).
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
        self.best_score_info = None  # for reference (Score-based)

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
        if res["cycle_acc"] > self.best_acc:  # original code: if best_acc < acc
            self.best_acc = res["cycle_acc"]
            self.best_info = info
            self.dirpath.mkdir(parents=True, exist_ok=True)
            model = getattr(pl_module, self.model_attr, pl_module)
            atomic_torch_save({"model": model.state_dict(), **info}, self.dirpath / self.filename)
            atomic_json_dump({**info, "best_score_info": self.best_score_info}, self.dirpath / "best.json")
            logger.info("Best (cycle acc) updated: epoch %d acc %.2f | Sp %.2f Se %.2f Score %.2f",
                        epoch, res["cycle_acc"], res["S_p"], res["S_e"], res["icbhi_score"])
        elif self.best_info is not None:
            # update only the Score-best reference record
            atomic_json_dump({**self.best_info, "best_score_info": self.best_score_info}, self.dirpath / "best.json")
