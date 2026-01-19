# rsc_toolkit/datasets//dataset.py
from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Union, Mapping
import pathlib
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset

from .icbhi.processing import load_audio, slice_audio

logger = logging.getLogger(__name__)


class IndexDataset(Dataset):
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
        sampling_rate: int = 10000,
        sample_level: Optional[Literal["cycle", "clip"]] = None,
        split: Optional[str] = None,
        split_file: Optional[Union[str, pathlib.Path]] = None,
        return_meta: bool = True,
        fade: bool = True,
        audio_preprocessor: Optional[Any] = None,
    ):
        self.index_csv = pathlib.Path(index_csv).expanduser().resolve()
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index csv not found: {self.index_csv}")

        self.df = pd.read_csv(self.index_csv)

        # If your SampleIndex.to_csv includes data_root somewhere else, you can load it.
        # Here we accept data_root explicitly, otherwise infer from index location or recording paths.
        self.data_root = pathlib.Path(data_root).expanduser().resolve() if data_root else None
        self.sampling_rate = int(sampling_rate)
        self.return_meta = bool(return_meta)
        self.fade = bool(fade)
        self.audio_preprocessor = audio_preprocessor

        if sample_level is None:
            # infer from first row
            if "sample_level" in self.df.columns and len(self.df) > 0:
                sample_level = str(self.df.iloc[0]["sample_level"])  # type: ignore[assignment]
            else:
                sample_level = "cycle"
        if sample_level not in ("cycle", "clip"):
            raise ValueError(f"Invalid sample_level: {sample_level}")

        self.sample_level: Literal["cycle", "clip"] = sample_level  # type: ignore[assignment] # TODO use enum instead
        self.split: str = split

        # Basic schema expectations
        required = {"sample_id", "recording_path", "sample_level", "t0", "t1", "label", "meta"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Index CSV missing columns: {sorted(missing)}")

        if split is not None:
            split_map = self._load_official_split_map(split_file)
            self._filter_df_by_split(split_map, split)

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

    @staticmethod
    def _default_split_file() -> pathlib.Path: # TODO not good to provide default
        return (pathlib.Path(__file__).parent / "data" / "official_split.txt").resolve()

    def _load_official_split_map(
        self, split_file: Optional[Union[str, pathlib.Path]]
    ) -> Mapping[str, str]:
        """
        Returns mapping: recording_id -> split ("train" or "test").
        """
        path = pathlib.Path(split_file).expanduser().resolve() if split_file else self._default_split_file()
        if not path.exists():   
            raise FileNotFoundError(f"Split file not found: {path}")

        mapping: Dict[str, str] = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            recording_id, split = parts[0], parts[1].lower()
            mapping[recording_id] = split
        return mapping

    def _filter_df_by_split(self, split_map: Mapping[str, str], split: str) -> None:
        def get_recording_id(recording_path: str) -> str:
            return pathlib.Path(str(recording_path)).stem

        before = int(len(self.df))
        recording_ids = self.df["recording_path"].map(get_recording_id)
        unassigned = int((~recording_ids.isin(list(split_map.keys()))).sum())
        mask = recording_ids.map(lambda rid: split_map.get(rid) == split)
        self.df = self.df.loc[mask].reset_index(drop=True)
        after = int(len(self.df))
        logger.info(
            "IndexDataset split=%s filtered %d -> %d samples (unassigned recordings=%d)",
            split,
            before,
            after,
            unassigned,
        )
        if len(self.df) == 0:
            raise ValueError(f"No data found for split {split}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]

        sample_id = str(row["sample_id"])
        recording_path = str(row["recording_path"])
        wav_path = self._resolve_recording_path(recording_path)
        if not wav_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {wav_path} (from recording_path={recording_path!r}, data_root={self.data_root})"
            )

        # Load audio. Resample + Fade here. Previously it was done to each cycle, why?
        wav = load_audio(wav_path, target_sampling_rate=self.sampling_rate, fade=self.fade)

        level = str(row["sample_level"])
        if level == "cycle":
            t0 = float(row["t0"])
            t1 = float(row["t1"])
            audio = slice_audio(wav, t0=t0, t1=t1, sampling_rate=self.sampling_rate)
            if self.audio_preprocessor is not None:
                audio = self.audio_preprocessor(audio)
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


class IndexFeatureDataset(Dataset):

    def __init__(self, index_ds, feature_processor):
        self.index_ds = index_ds
        self.fp = feature_processor

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, i):
        item = self.index_ds[i]
        audio = item["audio"].squeeze()
        feats = self.fp.make_features_cached(audio, sample_id=item["sample_id"])
        return feats, item["label"], item["sample_id"]
