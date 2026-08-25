"""
EpochResumeCheckpoint + ICBHIBestScoreCheckpoint 의 재개 동작.

시나리오: 4 epoch 짜리 학습이 2 epoch 뒤 죽었다 → 같은 설정으로 다시 띄우면
last.ckpt 에서 epoch 3 부터 이어가고, best 기준(이전 구간의 최고 Score)도 이어진다.
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
    """epoch 마다 정해진 가짜 Score 를 로그한다 (epoch 2 가 최고, 이후 하락)."""

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

    # 1) 2 epoch 만 돌고 "죽은" 것으로 가정 (정상 종료 삭제를 끄고 last.ckpt 를 남긴다)
    m1 = _Toy()
    _trainer(tmp_path, max_epochs=2, delete_on_finish=False).fit(m1, train, val)
    assert m1.epochs_trained == [0, 1]
    assert (tmp_path / "last.ckpt").exists()
    best_before = json.load(open(tmp_path / "best.json"))
    assert best_before["epoch"] == 2 and best_before["icbhi_score"] == 70.0

    # 2) 같은 설정으로 재발사 → epoch 3·4 만 돈다
    m2 = _Toy()
    t2 = _trainer(tmp_path, max_epochs=4)
    ckpt = find_resume_path(t2)
    assert ckpt == tmp_path / "last.ckpt"
    t2.fit(m2, train, val, ckpt_path=str(ckpt))
    assert m2.epochs_trained == [2, 3]

    # best 는 재개 전 구간(epoch 2, 70.0)이 유지돼야 한다 — 60/55 로 덮이면 프로토콜 위반
    best_after = json.load(open(tmp_path / "best.json"))
    assert best_after == best_before

    # 3) 정상 종료했으니 재개 파일은 지워진다
    assert not (tmp_path / "last.ckpt").exists()
    assert find_resume_path(t2) is None


def test_fresh_run_has_nothing_to_resume(tmp_path: pathlib.Path):
    assert find_resume_path(_trainer(tmp_path, max_epochs=1)) is None
