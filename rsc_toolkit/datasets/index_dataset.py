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
        # True: 원 sr 로 [t0,t1) 구간만 잘라 그 구간을 리샘플 (SCL-metadata 계열 dataset.py —
        #       torchaudio.load(frame_offset, num_frames) 후 Resample). 녹음 전체 리샘플 후 자르는 것과
        #       경계 샘플이 미세하게 다르다. fade 는 적용하지 않는다(전처리기가 담당).
        self.resample_after_slice = bool(resample_after_slice)

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

    # 인덱스 CSV 옆에 두는 split 파일 이름.
    # sprsound·hflungv1 processor 가 이미 이 이름으로 index_dir 에 써 놓는다.
    SPLIT_FILE_NAME = "official_split.txt"

    def _sibling_split_file(self) -> pathlib.Path:
        return (self.index_csv.parent / self.SPLIT_FILE_NAME).resolve()

    def _split_column_has_values(self) -> bool:
        """
        split 컬럼이 실제로 쓸 수 있는 값을 갖고 있는지 본다.

        ICBHI 인덱스는 split 컬럼이 있지만 전 행이 비어 있다. 컬럼 존재만 보고
        이 경로를 타면 매칭이 0건이라 `No data found for split ...` 로 죽는다.
        """
        if "split" not in self.df.columns:
            return False
        # 결측을 먼저 떨궈야 한다. pandas 3 에서 astype(str) 은 결측을 문자열
        # "nan" 으로 바꾸지 않고 결측인 채로 두므로, 문자열 비교로는 거를 수 없다.
        values = self.df["split"].dropna().astype(str).str.strip()
        return bool((values != "").any())

    def _apply_split(self, split: str, split_file: Optional[Union[str, pathlib.Path]]) -> None:
        """
        split 을 적용한다. 우선순위:
          1. 명시적으로 준 split_file
          2. 값이 들어 있는 split 컬럼
          3. 인덱스 CSV 옆의 official_split.txt
        """
        if split_file is not None:
            self._filter_df_by_split(self._load_official_split_map(split_file), split)
            return

        if self._split_column_has_values():
            self._filter_df_by_split_column(split)
            return

        sibling = self._sibling_split_file()
        if sibling.exists():
            logger.info("split 컬럼이 비어 있어 %s 를 사용한다.", sibling)
            self._filter_df_by_split(self._load_official_split_map(sibling), split)
            return

        raise ValueError(
            f"split={split!r} 을 적용할 수 없다. split_file 을 직접 주거나, "
            f"인덱스에 split 값이 들어 있거나, {sibling} 가 있어야 한다."
        )

    def _load_official_split_map(
        self, split_file: Union[str, pathlib.Path]
    ) -> Mapping[str, str]:
        """
        Returns mapping: recording_id -> split ("train" or "test").
        """
        path = pathlib.Path(split_file).expanduser().resolve()
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
            wav_native, sr = torchaudio.load(str(wav_path))
            audio = slice_audio(wav_native, t0=t0, t1=t1, sampling_rate=sr)
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
        # sgscl 계열(DAT·SG-SCL)용: 아이템 dict → meta 라벨 정수.
        # 지정하면 4번째 원소로 meta 라벨을 돌려준다 (원본은 train 셋에서만 meta 반환).
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
