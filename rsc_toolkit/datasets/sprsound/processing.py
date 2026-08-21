from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import pathlib
import logging

import pandas as pd


logger = logging.getLogger(__name__)


DEVICE_TO_ID = {
    "Yunting2": 0,
}

LOCATION_TO_ID = {
    "p1": 0,
    "p2": 1,
    "p3": 2,
    "p4": 3,
}


class LabelEncoder:
    def __init__(self) -> None:
        self.label_to_id: Dict[str, int] = {}

    def encode(self, value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)

        label = str(value).strip()
        if not label:
            return None
        if label.isdigit():
            return int(label)
        if label not in self.label_to_id:
            self.label_to_id[label] = len(self.label_to_id)
        return self.label_to_id[label]


def load_metadata(metadata_path: pathlib.Path) -> pd.DataFrame:
    metadata_path = metadata_path.expanduser().resolve()
    df = pd.read_csv(metadata_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in df.columns}


def _pick_column(cols: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def normalize_recording_id(value: str) -> str:
    name = pathlib.Path(str(value)).name
    if "." in name:
        name = pathlib.Path(name).stem
    return name


def normalize_split(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    split = str(value).strip().lower()
    if not split:
        return None
    if split in {"valid", "val", "dev", "validation"}:
        return "test"
    return split


def infer_split_from_path(value: str) -> Optional[str]:
    parts = [p.lower() for p in pathlib.Path(str(value)).parts]
    for part in parts:
        if "classification" not in part:
            continue
        if part.startswith("train") or part == "train":
            return "train"
        if part.startswith(("valid", "val", "test")) or part in {"valid", "val", "test"}:
            return "test"
    return None


def resolve_recording_path(
    *,
    data_root: pathlib.Path,
    filename: str,
    split: Optional[str],
) -> Optional[pathlib.Path]:
    cleaned = str(filename).strip()
    p = pathlib.Path(cleaned)

    if p.is_absolute() and p.exists():
        return p

    base = normalize_recording_id(p.name)

    candidates = [
        p,
        pathlib.Path(p.name),
        pathlib.Path(base),
        pathlib.Path(f"{base}.wav"),
    ]

    for candidate in candidates:
        if (data_root / candidate).exists():
            return (data_root / candidate).resolve()

    train_dir = data_root / "train_classification_wav"
    valid_dir = data_root / "valid_classification_wav"
    valid_2022_dir = valid_dir / "2022"

    def _rglob_first(root: pathlib.Path, patterns: Iterable[str]) -> Optional[pathlib.Path]:
        for pattern in patterns:
            match = next(root.rglob(pattern), None)
            if match is not None:
                return match.resolve()
        return None

    rglob_patterns = [f"{base}.wav", base]

    if split == "train":
        for candidate in candidates:
            if (train_dir / candidate).exists():
                return (train_dir / candidate).resolve()
        match = _rglob_first(train_dir, rglob_patterns)
        if match is not None:
            return match
    elif split in {"test", "valid"}:
        for candidate in candidates:
            if (valid_2022_dir / candidate).exists():
                return (valid_2022_dir / candidate).resolve()
            if (valid_dir / candidate).exists():
                return (valid_dir / candidate).resolve()
        match = _rglob_first(valid_2022_dir, rglob_patterns)
        if match is not None:
            return match
        match = _rglob_first(valid_dir, rglob_patterns)
        if match is not None:
            return match

    for sub in (train_dir, valid_2022_dir, valid_dir):
        for candidate in candidates:
            if (sub / candidate).exists():
                return (sub / candidate).resolve()
        match = _rglob_first(sub, rglob_patterns)
        if match is not None:
            return match

    return None


def coerce_number(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        val = float(value)
    except Exception:
        return value
    if val.is_integer():
        return int(val)
    return val
