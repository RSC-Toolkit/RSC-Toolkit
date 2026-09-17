"""
For GaP-Aug reproduction: records the test metrics at the end of training (final iterate) in best.json as final_*.

The paper does not state which model is evaluated after 20,000 iterations, so the metrics of the last validation are kept alongside the
toolkit default best-on-test (ICBHIBestScoreCheckpoint, no Se gate), allowing both protocols to be compared.
acc@1 (secondary paper-reported value: Table 1 Acc 69.01) is also kept as final_acc@1 (the best entry in best.json has no acc).
Must come after ICBHIBestScoreCheckpoint in the callback list (best.json must exist at on_fit_end for the merge).
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
        logger.info("Recorded final-iterate metrics → %s: %s", self.path, self._last)
