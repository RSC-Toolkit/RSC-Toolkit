"""
Resume behavior of EpochResumeCheckpoint + ICBHIBestScoreCheckpoint.

Scenario: a 4-epoch run dies after 2 epochs -> relaunching with the same settings
continues from epoch 3 using last.ckpt, and the best criterion (best Score of the earlier segment) carries over too.
"""
import json
import pathlib

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rsc_toolkit.trainer.callbacks.icbhi_best_checkpoint import ICBHIBestScoreCheckpoint
from rsc_toolkit.trainer.callbacks.resume_checkpoint import EpochResumeCheckpoint, find_resume_path


class _Toy(L.LightningModule):
    """Logs a fixed fake Score per epoch (epoch 2 is best, then it drops)."""

    SCORES = [50.0, 70.0, 60.0, 55.0]

    def __init__(self):
        super().__init__()
        self.rsc_model = nn.Linear(4, 2)
        self.epochs_trained = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.cross_entropy(self.rsc_model(x), y)

    def on_train_epoch_end(self):
        self.epochs_trained.append(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        s = self.SCORES[self.current_epoch]
        self.log_dict({"icbhi_score": s, "S_e": 40.0, "S_p": 80.0})

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loaders():
    x, y = torch.randn(16, 4), torch.randint(0, 2, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8), DataLoader(TensorDataset(x, y), batch_size=8)


def _trainer(tmp, max_epochs, **kw):
    return L.Trainer(
        max_epochs=max_epochs, accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[
            ICBHIBestScoreCheckpoint(dirpath=str(tmp)),
            EpochResumeCheckpoint(dirpath=str(tmp), **kw),
        ],
    )


def test_interrupted_run_resumes_from_last_epoch_and_keeps_best(tmp_path: pathlib.Path):
    train, val = _loaders()

    # 1) assume the run "died" after only 2 epochs (disable deletion on normal exit so last.ckpt remains)
    m1 = _Toy()
    _trainer(tmp_path, max_epochs=2, delete_on_finish=False).fit(m1, train, val)
    assert m1.epochs_trained == [0, 1]
    assert (tmp_path / "last.ckpt").exists()
    best_before = json.load(open(tmp_path / "best.json"))
    assert best_before["epoch"] == 2 and best_before["icbhi_score"] == 70.0

    # 2) relaunch with the same settings -> only epochs 3 and 4 run
    m2 = _Toy()
    t2 = _trainer(tmp_path, max_epochs=4)
    ckpt = find_resume_path(t2)
    assert ckpt == tmp_path / "last.ckpt"
    t2.fit(m2, train, val, ckpt_path=str(ckpt))
    assert m2.epochs_trained == [2, 3]

    # best must keep the pre-resume segment (epoch 2, 70.0) — being overwritten by 60/55 would violate the protocol
    best_after = json.load(open(tmp_path / "best.json"))
    assert best_after == best_before

    # 3) normal exit, so the resume file is deleted
    assert not (tmp_path / "last.ckpt").exists()
    assert find_resume_path(t2) is None


def test_fresh_run_has_nothing_to_resume(tmp_path: pathlib.Path):
    assert find_resume_path(_trainer(tmp_path, max_epochs=1)) is None
