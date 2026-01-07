# rsc_toolkit/datasets/icbhi/dataset.py
from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Union
import pathlib

import pandas as pd
import torch
from torch.utils.data import Dataset

from .processing import load_audio, slice_audio


class ICBHIIndexDataset(Dataset):
    """
    Torch-style dataset that consumes an index CSV produced by the processor.

    - sample_level="cycle": loads wav, slices [t0,t1]
    - sample_level="clip": loads full wav

    Returns a dict by default:
      {
        "audio": Tensor [C, T],
        "label": Optional[int],
        "sample_id": str,
        "meta": dict,
        ...
      }
    """

    def __init__(
        self,
        index_csv: Union[str, pathlib.Path],
        data_root: Optional[Union[str, pathlib.Path]] = None,
        sample_rate: int = 10000,
        sample_level: Optional[Literal["cycle", "clip"]] = None,
        return_meta: bool = True,
        fade: bool = True,
    ):
        self.index_csv = pathlib.Path(index_csv).expanduser().resolve()
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index csv not found: {self.index_csv}")

        self.df = pd.read_csv(self.index_csv)

        # If your SampleIndex.to_csv includes data_root somewhere else, you can load it.
        # Here we accept data_root explicitly, otherwise infer from index location or recording paths.
        self.data_root = pathlib.Path(data_root).expanduser().resolve() if data_root else None
        self.sample_rate = int(sample_rate)
        self.return_meta = bool(return_meta)
        self.fade = bool(fade)

        if sample_level is None:
            # infer from first row
            if "sample_level" in self.df.columns and len(self.df) > 0:
                sample_level = str(self.df.iloc[0]["sample_level"])  # type: ignore[assignment]
            else:
                sample_level = "cycle"
        if sample_level not in ("cycle", "clip"):
            raise ValueError(f"Invalid sample_level: {sample_level}")

        self.sample_level: Literal["cycle", "clip"] = sample_level  # type: ignore[assignment]

        # Basic schema expectations
        required = {"sample_id", "recording_path", "sample_level", "t0", "t1", "label", "meta"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Index CSV missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return int(len(self.df))

    def _resolve_recording_path(self, recording_path: str) -> pathlib.Path:
        p = pathlib.Path(recording_path)
        if p.is_absolute():
            return p
        if self.data_root is None:
            # assume relative to index location's parent as a fallback
            return (self.index_csv.parent / p).resolve()
        return (self.data_root / p).resolve()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]

        sample_id = str(row["sample_id"])
        recording_path = str(row["recording_path"])
        wav_path = self._resolve_recording_path(recording_path)

        # Load audio (resample + optional fade)
        wav = load_audio(wav_path, target_sample_rate=self.sample_rate, fade=self.fade)

        level = str(row["sample_level"])
        if level == "cycle":
            t0 = float(row["t0"])
            t1 = float(row["t1"])
            audio = slice_audio(wav, t0=t0, t1=t1, sample_rate=self.sample_rate)
        elif level == "clip":
            audio = wav
            t0 = None
            t1 = None
        else:
            raise ValueError(f"Unexpected sample_level in index: {level}")

        label = row["label"]
        label_out: Optional[int] = None if pd.isna(label) else int(label)

        out: Dict[str, Any] = {
            "audio": audio,
            "label": label_out,
            "sample_id": sample_id,
            "recording_path": recording_path,
            "sample_level": level,
            "t0": t0,
            "t1": t1,
        }

        if self.return_meta:
            # meta was written as dict by SampleIndex; depending on implementation,
            # it may be a JSON string in CSV.
            meta_val = row["meta"]
            if isinstance(meta_val, str):
                # safest: try json decode
                import json
                try:
                    meta_val = json.loads(meta_val)
                except Exception:
                    meta_val = {"_raw": meta_val}
            out["meta"] = meta_val

        return out
