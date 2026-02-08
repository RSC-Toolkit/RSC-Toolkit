from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import pathlib
import logging

import pandas as pd


logger = logging.getLogger(__name__)


DEVICE_TO_ID = {
    "Litt3200": 0,
    "HF_Type-1": 1,
}

LOCATION_TO_ID = {
    "L1": 0,
    "L2": 1,
    "L4": 2,
    "L5": 3,
    "L6": 4,
    "L8": 5,
    "unknown": 6,
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
    if name.endswith("_label"):
        name = name[:-6]
    if "." in name:
        name = pathlib.Path(name).stem
    return name


def resolve_recording_path(
    *,
    data_root: pathlib.Path,
    filename: str,
    split: Optional[str],
) -> Optional[pathlib.Path]:
    p = pathlib.Path(str(filename))

    if p.is_absolute() and p.exists():
        return p

    base = normalize_recording_id(p.name)
    candidates: List[pathlib.Path] = []
    if p.parent and str(p.parent) != ".":
        candidates.append(p)
        candidates.append(p.parent / base)
        candidates.append(p.parent / f"{base}.wav")
    candidates.append(pathlib.Path(p.name))
    candidates.append(pathlib.Path(base))
    candidates.append(pathlib.Path(f"{base}.wav"))

    for candidate in candidates:
        if (data_root / candidate).exists():
            return (data_root / candidate).resolve()

    if split:
        split_dir = data_root / str(split).lower()
        for candidate in candidates:
            if (split_dir / candidate).exists():
                return (split_dir / candidate).resolve()

    # Try common train/test fallbacks if split not provided
    for sub in ("train", "test"):
        candidate = data_root / sub / p
        if candidate.exists():
            return candidate.resolve()
        for name in (p.name, base, f"{base}.wav"):
            candidate = data_root / sub / name
            if candidate.exists():
                return candidate.resolve()

    return None


def find_annotation_path(recording_path: pathlib.Path) -> Optional[pathlib.Path]:
    stem = recording_path.stem
    candidates = [
        recording_path.with_suffix(".txt"),
        recording_path.with_suffix(".csv"),
        recording_path.parent / "annotation" / f"{stem}.txt",
        recording_path.parent / "annotation" / f"{stem}.csv",
        recording_path.parent / "annotations" / f"{stem}.txt",
        recording_path.parent / "annotations" / f"{stem}.csv",
        recording_path.parent.parent / "annotation" / f"{stem}.txt",
        recording_path.parent.parent / "annotation" / f"{stem}.csv",
        recording_path.parent.parent / "annotations" / f"{stem}.txt",
        recording_path.parent.parent / "annotations" / f"{stem}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _coerce_float(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _parse_cycles_from_df(
    df: pd.DataFrame,
    *,
    label_encoder: LabelEncoder,
    default_label: Optional[int],
) -> List[Tuple[float, float, Optional[int]]]:
    cols = _normalize_columns(df)

    start_col = _pick_column(cols, ["start", "begin", "t0", "onset"])
    end_col = _pick_column(cols, ["end", "stop", "t1", "offset"])
    label_col = _pick_column(cols, ["label", "class", "diagnosis", "disease", "target", "y"])
    crackle_col = _pick_column(cols, ["crackle"])
    wheeze_col = _pick_column(cols, ["wheeze"])

    if start_col is None or end_col is None:
        # If no headers, try positional fallback
        if df.shape[1] >= 2:
            start_col = df.columns[0]
            end_col = df.columns[1]
            if label_col is None and df.shape[1] >= 3:
                label_col = df.columns[2]

    out: List[Tuple[float, float, Optional[int]]] = []
    if start_col is None or end_col is None:
        return out

    for _, row in df.iterrows():
        t0 = _coerce_float(row[start_col])
        t1 = _coerce_float(row[end_col])
        if t0 is None or t1 is None:
            continue

        label: Optional[int] = default_label
        if crackle_col and wheeze_col:
            try:
                crackle = int(row[crackle_col])
                wheeze = int(row[wheeze_col])
                label = crackle + 2 * wheeze
            except Exception:
                label = default_label
        elif label_col:
            label = label_encoder.encode(row[label_col])
            if label is None:
                label = default_label

        out.append((t0, t1, label))
    return out


def parse_cycle_annotations(
    annotation_path: pathlib.Path,
    *,
    label_encoder: LabelEncoder,
    default_label: Optional[int],
) -> List[Tuple[float, float, Optional[int]]]:
    try:
        df = pd.read_csv(annotation_path, sep=None, engine="python")
    except Exception as exc:
        logger.warning("Failed to read annotations (%s): %s", annotation_path, exc)
        return []

    # If headers look like integers, treat as headerless
    if all(isinstance(c, int) or str(c).isdigit() for c in df.columns):
        df = pd.read_csv(annotation_path, header=None)

    return _parse_cycles_from_df(
        df,
        label_encoder=label_encoder,
        default_label=default_label,
    )
