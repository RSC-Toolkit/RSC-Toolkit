"""AMSDEpochCallback — resuming works only if the per-sample buffer size correction happens **before** last.ckpt is restored (setup hook).

Previously the correction ran in on_fit_start (after restore), so resuming the last.ckpt of a SPRSound (6,656) run, whose size differs from the
constructor default (ICBHI 4,142), always failed with a strict-load size mismatch. Instead of a real AST, a small model with the same buffer
contract (all_predictions/ids_masked/current_epoch + ensure_buffers) pins the Lightning hook order.
"""
from __future__ import annotations

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rsc_toolkit.trainer.callbacks.amsd_epoch import AMSDEpochCallback

N_DEFAULT, N_DATA = 4, 6


class _BufferedModel(nn.Module):
    def __init__(self, num_train_samples: int = N_DEFAULT):
        super().__init__()
        self.n_cls, self.num_masked = 2, 1
        self.lin = nn.Linear(2, 2)
        self.register_buffer("all_predictions", torch.zeros(num_train_samples, self.n_cls))
        self.register_buffer("ids_masked", torch.zeros(num_train_samples, self.num_masked, dtype=torch.long))
        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long))

    def ensure_buffers(self, num_train_samples: int) -> None:   # same contract as ast_amsd.ASTAMSD.ensure_buffers
        n = int(num_train_samples)
        if self.all_predictions.shape[0] == n:
            return
        dev = self.all_predictions.device
        self.all_predictions = torch.zeros(n, self.n_cls, device=dev)
        self.ids_masked = torch.zeros(n, self.num_masked, dtype=torch.long, device=dev)


class _PL(L.LightningModule):
    def __init__(self, num_train_samples: int = N_DEFAULT):
        super().__init__()
        self.rsc_model = _BufferedModel(num_train_samples)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        return self.rsc_model.lin(x).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _DM(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_ds = None

    def setup(self, stage=None):
        self.train_ds = TensorDataset(torch.randn(N_DATA, 2), torch.zeros(N_DATA, dtype=torch.long))

    @property
    def num_train_samples(self):
        return None if self.train_ds is None else len(self.train_ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=3)


def _trainer(max_epochs: int):
    return L.Trainer(max_epochs=max_epochs, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False, callbacks=[AMSDEpochCallback()])


def test_buffers_follow_datamodule_and_resume_after_size_change(tmp_path):
    torch.manual_seed(0)
    pl = _PL()
    _trainer(1).fit(pl, datamodule=_DM())
    assert tuple(pl.rsc_model.all_predictions.shape) == (N_DATA, 2)          # corrected 4 -> 6
    ckpt = tmp_path / "last.ckpt"
    trainer = _trainer(1)
    trainer.fit(pl, datamodule=_DM())            # max_epochs reached -> exits immediately; keeps a trainer for saving state
    trainer.save_checkpoint(ckpt)
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]["rsc_model.all_predictions"].shape[0] == N_DATA

    # new-process scenario: resume a ckpt with size-6 buffers into a model built with the constructor default size (4) — strict restore succeeds thanks to the setup-hook correction
    fresh = _PL()
    assert tuple(fresh.rsc_model.all_predictions.shape) == (N_DEFAULT, 2)
    _trainer(2).fit(fresh, datamodule=_DM(), ckpt_path=str(ckpt))
    assert tuple(fresh.rsc_model.all_predictions.shape) == (N_DATA, 2)
    assert int(fresh.rsc_model.current_epoch) == 1                            # ran the second epoch after resuming


def test_on_fit_start_marks_callback_attached():
    pl = _PL()
    _trainer(1).fit(pl, datamodule=_DM())
    assert getattr(pl.rsc_model, "_epoch_callback_attached", False) is True
