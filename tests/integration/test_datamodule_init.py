from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule


class DummyIndexDataset(Dataset):
    """
    Minimal index dataset used to unit-test IndexDataModule.

    Matches the expected constructor shape of index datasets:
      (index_csv, data_root, sampling_rate, sample_level, split, split_file?)
    """

    def __init__(
        self,
        index_csv: Path,
        data_root: Path,
        sampling_rate: int,
        sample_level: str,
        split: str,
        split_file: Optional[Path] = None,
    ):
        _ = (data_root, sampling_rate, sample_level, split_file)
        self.df = pd.read_csv(Path(index_csv))
        self.df = self.df.loc[self.df["split"].astype(str) == str(split)].reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        # Only used if someone enables features in this unit test.
        return {
            "audio": torch.zeros(1, 16),
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
        }


def test_index_datamodule_builds_splits(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    index_dir = out_dir / "dataset" / "DUMMY" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"sample_id": "a", "label": 0, "split": "train"},
            {"sample_id": "b", "label": 1, "split": "train"},
            {"sample_id": "c", "label": 0, "split": "test"},
        ]
    )
    df.to_csv(index_dir / "index_cycle.csv", index=False)

    dm = IndexDataModule(
        dataset_name="DUMMY",
        index_dataset_cls=DummyIndexDataset,
        project_root=".",
        raw_data_directory=str(tmp_path / "raw"),
        outputs_directory=str(out_dir),
        split_file=None,
        sampling_rate=16_000,
        sample_level="cycle",
        features={"enabled": False},
    )
    dm.setup("fit")

    assert dm.train_ds is not None
    assert dm.val_ds is not None
    assert len(dm.train_ds) == 2
    assert len(dm.val_ds) == 1


def test_index_datamodule_supports_split_lists(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    index_dir = out_dir / "dataset" / "DUMMY" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Simulate arbitrary split labels (including numbers).
    df = pd.DataFrame(
        [
            {"sample_id": "a", "label": 0, "split": 0},
            {"sample_id": "b", "label": 1, "split": 1},
            {"sample_id": "c", "label": 0, "split": 2},
            {"sample_id": "d", "label": 1, "split": 2},
        ]
    )
    df.to_csv(index_dir / "index_cycle.csv", index=False)

    dm = IndexDataModule(
        dataset_name="DUMMY",
        index_dataset_cls=DummyIndexDataset,
        project_root=".",
        raw_data_directory=str(tmp_path / "raw"),
        outputs_directory=str(out_dir),
        split_file=None,
        sampling_rate=16_000,
        sample_level="cycle",
        train_splits=[0, 1],
        val_splits=[2],
        test_splits=[2],
        features={"enabled": False},
    )
    dm.setup("fit")
    dm.setup("test")

    assert dm.train_ds is not None
    assert dm.val_ds is not None
    assert dm.test_ds is not None
    assert len(dm.train_ds) == 2
    assert len(dm.val_ds) == 2
    assert len(dm.test_ds) == 2
