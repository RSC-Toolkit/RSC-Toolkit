"""
BTS 논문(kaen2891/bts)이 배포한 사전 전처리 ICBHI 텐서(training.pt / test.pt) 로더.

원본 util/icbhi_dataset.py 는 첫 실행 때 48 kHz 로 리샘플·8초 cut/pad 한 cycle 들을
HF ClapProcessor 로 log-mel(1, 1001, 64) 로 바꿔 .pt 로 저장하고 이후엔 그것만 읽는다.
저장소에는 이 .pt 만 배포되고(전처리 스크립트 없음), 같은 파일을 팀 논문 BTS-CARD 도
쓴다. 논문 수치와 같은 입력을 보장하려고 이 파일을 그대로 읽는 경로를 둔다.

항목 = (log-mel Tensor(1, 1001, 64), label int, meta Tensor(7,) [age, sex, adult_BMI,
child_weight, child_height, chest_location_id, device_id]).
배치 = (x (B, 1, 1001, 64), y (B,), meta (B, 7)) — 검증 배치에도 meta 가 붙는다
(BTS 는 평가 때도 텍스트 설명을 쓴다; 원본 validate 와 동일).
"""
from __future__ import annotations

import logging
import pathlib
from typing import List, Optional, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BTSPrecomputedDataset(Dataset):
    def __init__(self, path):
        self.path = pathlib.Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"BTS 전처리 텐서가 없다: {self.path} (README 의 Google Drive 링크)")
        self.items = torch.load(self.path, map_location="cpu", weights_only=False)
        logger.info("BTS 전처리 텐서 로드: %s (%d cycles)", self.path, len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> Tuple[torch.Tensor, int, torch.Tensor]:
        feat, label, meta = self.items[i]
        return feat, int(label), meta.float()


def bts_collate(batch: List[Tuple[torch.Tensor, int, torch.Tensor]]):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    meta = torch.stack([b[2] for b in batch], dim=0)
    return x, y, meta


class BTSPrecomputedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str = "pretrained_models/upstream/bts/training.pt",
        test_path: str = "pretrained_models/upstream/bts/test.pt",
        batch_size: int = 8,
        num_workers: int = 8,
        shuffle_train: bool = True,
        drop_last_train: bool = True,   # 원본 train loader drop_last=True
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle_train = bool(shuffle_train)
        self.drop_last_train = bool(drop_last_train)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.train_ds: Optional[BTSPrecomputedDataset] = None
        self.test_ds: Optional[BTSPrecomputedDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_ds is None:
            self.train_ds = BTSPrecomputedDataset(self.train_path)
        if self.test_ds is None:
            self.test_ds = BTSPrecomputedDataset(self.test_path)

    def _loader(self, ds, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=self.num_workers, collate_fn=bts_collate, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=self.shuffle_train, drop_last=self.drop_last_train)

    def val_dataloader(self) -> DataLoader:
        # 논문 프로토콜: 검증셋 없이 매 epoch 공식 test 평가 (best-on-test)
        return self._loader(self.test_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False, drop_last=False)
