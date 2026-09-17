"""Saves the model state_dict at the end of training, for protocols that hand over the last-epoch state without evaluation (e.g. SCL/M-SCL pretraining)."""
import logging
import pathlib

import lightning as L

from rsc_toolkit.trainer.callbacks.atomic_io import atomic_torch_save

logger = logging.getLogger(__name__)


class FinalStateCheckpoint(L.Callback):
    def __init__(self, dirpath: str, filename: str = "final_state.pth", model_attr: str = "rsc_model"):
        super().__init__()
        self.dirpath, self.filename, self.model_attr = pathlib.Path(dirpath), filename, model_attr

    def on_train_end(self, trainer, pl_module):
        model = getattr(pl_module, self.model_attr, pl_module)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({"model": model.state_dict(), "epoch": trainer.current_epoch}, self.dirpath / self.filename)
        logger.info("Saved final state: %s (epoch %d)", self.dirpath / self.filename, trainer.current_epoch)
