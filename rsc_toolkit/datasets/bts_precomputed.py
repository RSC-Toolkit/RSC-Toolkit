"""
Loader for the preprocessed ICBHI tensors (training.pt / test.pt) released with the BTS paper (kaen2891/bts).

The original util/icbhi_dataset.py, on first run, resamples cycles to 48 kHz with an 8 s cut/pad,
converts them with the HF ClapProcessor to log-mel (1, 1001, 64), saves them as .pt and reads only those afterwards.
Only these .pt files are released in the repository (no preprocessing script), and the BTS-CARD paper uses
the same files. This path reads them as-is to guarantee the same inputs as the paper numbers.

Item = (log-mel Tensor(1, 1001, 64), label int, meta Tensor(7,) [age, sex, adult_BMI,
child_weight, child_height, chest_location_id, device_id]).
Batch = (x (B, 1, 1001, 64), y (B,), meta (B, 7)) -- validation batches also carry meta
(BTS uses the text description at evaluation too; same as the original validate).
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
            raise FileNotFoundError(f"BTS preprocessed tensor not found: {self.path} (Google Drive link in the README)")
        self.items = torch.load(self.path, map_location="cpu", weights_only=False)
        logger.info("Loaded BTS preprocessed tensor: %s (%d cycles)", self.path, len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> Tuple[torch.Tensor, int, torch.Tensor]:
        feat, label, meta = self.items[i]
        return feat, int(label), meta.float()


from rsc_toolkit.datasets import collate as _collate  # single collate implementation

# (x, y, meta7 stack)
bts_collate = _collate.stack_meta_collate


class BTSPrecomputedDataModule(L.LightningDataModule):
    meta_kind = "bts_meta7"  # batch[2] = meta7-shaped float vector (both train and val) -- bts/bts_card/audioclap

    def __init__(
        self,
        train_path: str = "pretrained_models/upstream/bts/training.pt",
        test_path: str = "pretrained_models/upstream/bts/test.pt",
        batch_size: int = 8,
        num_workers: int = 8,
        shuffle_train: bool = True,
        drop_last_train: bool = True,   # original train loader drop_last=True
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
        # Paper protocol: no validation set, official test evaluated every epoch (best-on-test)
        return self._loader(self.test_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False, drop_last=False)
