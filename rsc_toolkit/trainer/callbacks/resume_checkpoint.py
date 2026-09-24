"""
Per-epoch resume checkpoint: lets training continue from the last completed epoch even if the
process is killed by an external cause (e.g. a Slurm job ending).

At the end of every epoch (after that epoch's validation and best update), the full Lightning state
(model, optimizer, AMP scaler, loop progress, callback states) is saved to `last.ckpt`.
When the same run is launched again, train.py resumes automatically if this file exists (resume: auto).
The file is deleted on normal completion (the only output should be best.pth).

No effect on the paper protocol (numbers): only saving is added, and the only thing that differs on resume is
the RNG stream (SpecAugment masks, shuffle order), because Lightning does not restore the global RNG.
Not bit-identical to an uninterrupted run, but training from the same distribution.
"""
from __future__ import annotations

import logging
import os
import pathlib

import lightning as L

logger = logging.getLogger(__name__)


class EpochResumeCheckpoint(L.Callback):
    def __init__(self, dirpath: str, filename: str = "last.ckpt", delete_on_finish: bool = True):
        super().__init__()
        self.dirpath = pathlib.Path(dirpath)
        self.filename = filename
        self.delete_on_finish = bool(delete_on_finish)

    @property
    def last_path(self) -> pathlib.Path:
        return self.dirpath / self.filename

    def on_train_epoch_end(self, trainer, pl_module):
        # Validation (on_validation_end → best update) finishes before this hook.
        if trainer.sanity_checking:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        tmp = self.last_path.with_name(self.last_path.name + ".tmp")
        trainer.save_checkpoint(str(tmp))
        os.replace(tmp, self.last_path)  # if the process dies mid-save, the previous last.ckpt stays intact

    def on_fit_end(self, trainer, pl_module):
        # trainer.state.finished is set *after* this hook, so check whether the loop is done
        # (max_epochs reached or should_stop). This hook is not called on exception/interruption.
        if self.delete_on_finish and trainer.fit_loop.done and self.last_path.exists():
            self.last_path.unlink()
            logger.info("Training finished normally; deleted resume checkpoint %s", self.last_path)


def find_resume_path(trainer: L.Trainer) -> pathlib.Path | None:
    """Returns the path of last.ckpt if the trainer has an EpochResumeCheckpoint and that file exists."""
    for cb in trainer.callbacks:
        if isinstance(cb, EpochResumeCheckpoint) and cb.last_path.exists():
            return cb.last_path
    return None
