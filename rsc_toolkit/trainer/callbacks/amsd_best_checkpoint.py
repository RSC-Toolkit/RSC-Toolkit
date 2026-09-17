"""
Best selection for AMS-D: same protocol as ICBHIBestScoreCheckpoint (val = official test, highest Score && Se > min_se), except that
(1) the selection keys are configurable (default: the paper's binary metrics paper_score/paper_S_e/paper_S_p), and
(2) best.json records **both** the standard 4-class ICBHIScore (S_p/S_e/icbhi_score) and the binary metrics.

To select best by the standard formula, set score_key=icbhi_score se_key=S_e sp_key=S_p.
The official code's selection (best_acc/best_score ckpt on an 80/20 valid split inside train, test.py default epoch 45) is documented, not implemented.
"""
from __future__ import annotations

import logging
from typing import Sequence


from rsc_toolkit.trainer.callbacks.atomic_io import atomic_json_dump, atomic_torch_save

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
        min_se: float = -1.0,             # no Se gate in the original code
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
            # Fail fast for the same reason as the parent (ICBHIBestScoreCheckpoint): avoid silently finishing with no output.
            raise RuntimeError(
                f"best-selection metric '{self.score_key}' is not logged (available keys: {sorted(map(str, metrics))}). "
                f"Check that the metric is in the val list of the metrics config; "
                f"otherwise training finishes without best.pth/best.json."
            )
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
        atomic_torch_save({"model": model.state_dict(), **info}, path)
        atomic_json_dump(info, path.with_suffix(".json"))
        logger.info("Best updated (%s): epoch %d, %s=%.2f | standard Sp/Se/Score %s/%s/%s", self.score_key, info["epoch"], self.score_key, score,
                    info.get("S_p"), info.get("S_e"), info.get("icbhi_score"))
