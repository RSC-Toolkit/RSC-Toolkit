# Portions of this file are adapted from patch-mix_contrastive_learning (https://github.com/raymin0223/patch-mix_contrastive_learning); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Model selection of the ICBHI paper protocol: reproduces validate() and the saving logic of patch-mix main.py.

The original code has no validation set; it evaluates the official test split every epoch and saves the
model+classifier state when "highest Score && Se > 5" (reported numbers are effectively best-on-test).
Comparing against paper-reported values requires the same protocol, so it is reproduced as-is.
Assumes the official test split is attached to the val dataloader (val_splits: [test] in the config).

Reads "icbhi_score" / "S_e" / "S_p" logged by the ICBHIScore metric.
"""
import logging
import pathlib

import lightning as L

from rsc_toolkit.trainer.callbacks.atomic_io import atomic_json_dump, atomic_torch_save

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

    # The best criterion must carry over on resume (EpochResumeCheckpoint); otherwise the first epoch
    # after resume always overwrites best, breaking the protocol (highest Score over the whole run).
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
            # Only warning and skipping here would let a run with icbhi_score missing from the metrics config
            # finish with exit 0 but no best.pth/best.json; an unattended scheduler would treat it as success
            # and aggregation would silently drop that seed from the average.
            # Failing at the first validation avoids wasted GPU time and a silently reduced sample count.
            raise RuntimeError(
                f"best-selection metric '{self.score_key}' is not logged (available keys: {sorted(map(str, metrics))}). "
                f"Check that the metric is in the val list of the metrics config; "
                f"otherwise training finishes without best.pth/best.json."
            )
        score = float(metrics[self.score_key])
        se = float(metrics.get(self.se_key, float("nan")))
        sp = float(metrics.get(self.sp_key, float("nan")))

        # original code: if sc > best_acc[-1] and se > 5
        if not (score > self.best_score and se > self.min_se):
            return

        self.best_score = score
        self.best_info = {"epoch": trainer.current_epoch + 1, "S_p": sp, "S_e": se, "icbhi_score": score}

        self.dirpath.mkdir(parents=True, exist_ok=True)
        model = getattr(pl_module, self.model_attr, pl_module)
        path = self.dirpath / self.filename
        atomic_torch_save({"model": model.state_dict(), **self.best_info}, path)
        atomic_json_dump(self.best_info, path.with_suffix(".json"))
        logger.info("Best updated: epoch %d, Score %.2f (Sp %.2f / Se %.2f) -> %s",
                    self.best_info["epoch"], score, sp, se, path)

    def on_fit_end(self, trainer, pl_module):
        # Training finished without ever passing the gate (Se > min_se). This is a valid protocol outcome,
        # so no exception, but without best.json aggregation silently drops this seed.
        # Leave a visible warning for log-only monitoring.
        if self.best_info is None:
            logger.warning(
                "Training finished but best was never updated (gate %s > %.2f not met); "
                "%s/best.json does not exist. This run is excluded from aggregation.",
                self.se_key, self.min_se, self.dirpath,
            )
