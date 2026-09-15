"""AMSDEpochCallback — per-sample 버퍼 크기 보정이 last.ckpt 복원 **전**(setup 훅)에 일어나야 재개가 된다.

REVIEW 2026-09-10 §4/§5 P1: 예전엔 on_fit_start(복원 뒤) 에서 보정해, 생성자 기본 크기(ICBHI 4,142)와 다른
SPRSound(6,656) run 의 last.ckpt 재개가 strict load size mismatch 로 반드시 실패했다. 실제 AST 대신 같은 버퍼
계약(all_predictions/ids_masked/current_epoch + ensure_buffers)을 가진 작은 모델로 Lightning 훅 순서를 고정한다.
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

    def ensure_buffers(self, num_train_samples: int) -> None:   # ast_amsd.ASTAMSD.ensure_buffers 와 동일 계약
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
    assert tuple(pl.rsc_model.all_predictions.shape) == (N_DATA, 2)          # 4 → 6 보정
    ckpt = tmp_path / "last.ckpt"
    trainer = _trainer(1)
    trainer.fit(pl, datamodule=_DM())            # max_epochs 도달 → 즉시 종료, 상태 저장용 trainer 확보
    trainer.save_checkpoint(ckpt)
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]["rsc_model.all_predictions"].shape[0] == N_DATA

    # 새 프로세스 상황: 생성자 기본 크기(4)로 만든 모델에 6 크기 버퍼 ckpt 를 재개 — setup 훅 보정 덕에 strict 복원 성공
    fresh = _PL()
    assert tuple(fresh.rsc_model.all_predictions.shape) == (N_DEFAULT, 2)
    _trainer(2).fit(fresh, datamodule=_DM(), ckpt_path=str(ckpt))
    assert tuple(fresh.rsc_model.all_predictions.shape) == (N_DATA, 2)
    assert int(fresh.rsc_model.current_epoch) == 1                            # 재개 후 두 번째 epoch 를 돌았다


def test_on_fit_start_marks_callback_attached():
    pl = _PL()
    _trainer(1).fit(pl, datamodule=_DM())
    assert getattr(pl.rsc_model, "_epoch_callback_attached", False) is True
