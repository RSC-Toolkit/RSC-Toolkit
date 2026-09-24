# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
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
        resample_after_slice: bool = False,
        time_unit: str = "s",
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
        # True: crop only the [t0,t1) segment at the original sr and resample that segment (SCL-metadata family dataset.py --
        #       torchaudio.load(frame_offset, num_frames) then Resample). Boundary samples differ slightly from resampling
        #       the whole recording and then cropping. No fade is applied (the preprocessor handles it).
        self.resample_after_slice = bool(resample_after_slice)
        # "ms_int": SPRSound family (scl_icbhi2017 dataset.py:202) -- onsets/offsets are integer ms, so
        #          crop at start = ms * int(sr/1000) (t0 is stored in seconds and recovered as round(t0*1000)).
        if time_unit not in ("s", "ms_int"):
            raise ValueError(f"time_unit must be 's' or 'ms_int', got {time_unit!r}")
        self.time_unit = time_unit

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
        from rsc_toolkit.datasets.sample_index import REQUIRED_INDEX_COLUMNS

        missing = set(REQUIRED_INDEX_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"Index CSV missing columns: {sorted(missing)}")

        if split is not None:
            self._apply_split(split, split_file)

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

    # Name of the split file placed next to the index CSV.
    # The sprsound and hflungv1 processors write it to index_dir under this name.
    SPLIT_FILE_NAME = "official_split.txt"

    def _sibling_split_file(self) -> pathlib.Path:
        return (self.index_csv.parent / self.SPLIT_FILE_NAME).resolve()

    def _split_column_has_values(self) -> bool:
        """
        Check whether the split column holds usable values.

        ICBHI indexes have a split column, but every row is empty. Taking this path based only on
        column presence would match 0 rows and fail with `No data found for split ...`.
        """
        if "split" not in self.df.columns:
            return False
        # Missing values must be dropped first. In pandas 3, astype(str) keeps missing values as missing
        # instead of turning them into the string "nan", so a string comparison cannot filter them.
        values = self.df["split"].dropna().astype(str).str.strip()
        return bool((values != "").any())

    def _apply_split(self, split: str, split_file: Optional[Union[str, pathlib.Path]]) -> None:
        """
        Apply the split. Priority:
          1. an explicitly given split_file
          2. a split column with values
          3. official_split.txt next to the index CSV
        """
        if split_file is not None:
            self._filter_df_by_split(self._load_official_split_map(split_file), split)
            return

        if self._split_column_has_values():
            self._filter_df_by_split_column(split)
            return

        sibling = self._sibling_split_file()
        if sibling.exists():
            logger.info("split column is empty; using %s.", sibling)
            self._filter_df_by_split(self._load_official_split_map(sibling), split)
            return

        raise ValueError(
            f"Cannot apply split={split!r}. Pass split_file explicitly, "
            f"or the index must contain split values, or {sibling} must exist."
        )

    def _load_official_split_map(
        self, split_file: Union[str, pathlib.Path]
    ) -> Mapping[str, str]:
        """recording_id -> split ("train" or "test"). The single parser is rsc_toolkit.datasets.splits.read_split_file."""
        from rsc_toolkit.datasets.splits import read_split_file

        return read_split_file(split_file)

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

    def _filter_df_by_split_column(self, split: str) -> None:
        before = int(len(self.df))
        split_series = self.df["split"].astype(str).str.strip().str.lower()
        mask = split_series == str(split).strip().lower()
        self.df = self.df.loc[mask].reset_index(drop=True)
        after = int(len(self.df))
        logger.info(
            "IndexDataset split=%s filtered %d -> %d samples (split column)",
            split,
            before,
            after,
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

        level = str(row["sample_level"])
        if level == "cycle" and self.resample_after_slice:
            import torchaudio
            import torchaudio.transforms as T

            t0 = float(row["t0"])
            t1 = float(row["t1"])
            # Read only the segment, as in the original (scl_icbhi2017 dataset.py:93-94): frame_offset=int(t0*sr),
            # num_frames=int(t1*sr)-int(t0*sr). For WAV this yields the same samples as reading everything and cropping, and is much faster.
            sr = torchaudio.info(str(wav_path)).sample_rate
            if self.time_unit == "ms_int":
                k = int(sr / 1000)
                start, end = int(round(t0 * 1000)) * k, int(round(t1 * 1000)) * k
            else:
                start, end = int(t0 * sr), int(t1 * sr)
            audio, _ = torchaudio.load(str(wav_path), frame_offset=start, num_frames=max(end - start, 0))
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != self.sampling_rate:
                audio = T.Resample(sr, self.sampling_rate)(audio)
            if self.audio_preprocessor is not None:
                audio = self.audio_preprocessor(audio)
            wav = None
        else:
            # Load audio. Resample + Fade here. Previously it was done to each cycle, why?
            wav = load_audio(wav_path, target_sampling_rate=self.sampling_rate, fade=self.fade)

        if level == "cycle" and wav is not None:
            t0 = float(row["t0"])
            t1 = float(row["t1"])
            audio = slice_audio(wav, t0=t0, t1=t1, sampling_rate=self.sampling_rate)
            if self.audio_preprocessor is not None:
                audio = self.audio_preprocessor(audio)
        elif level == "cycle":
            pass
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

    def __init__(
        self,
        index_ds,
        feature_processor,
        feature_transform: Optional[Any] = None,
        meta_label_fn: Optional[Any] = None,
    ):
        self.index_ds = index_ds
        self.fp = feature_processor
        self.feature_transform = feature_transform
        # For the sgscl family (DAT, SG-SCL): item dict -> integer meta label.
        # When set, the meta label is returned as the 4th element (the original returns meta only for the train set).
        self.meta_label_fn = meta_label_fn

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, i):
        item = self.index_ds[i]
        audio = item["audio"].squeeze()
        feats = self.fp.make_features_cached(audio, sample_id=item["sample_id"])
        if self.feature_transform is not None:
            feats = self.feature_transform(feats)
        if self.meta_label_fn is not None:
            return feats, item["label"], item["sample_id"], int(self.meta_label_fn(item))
        return feats, item["label"], item["sample_id"]
