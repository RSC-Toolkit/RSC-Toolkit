"""
ICBHI data path for Domain Transfer (Wang & Wang, ICASSP 2022) -- **re-implementation from the paper description** (no code released, grade C).

Stated in the paper (Sec. 2.1, Sec. 3.2):
- All audio is **resampled to 8 kHz** and unified to **about 8 s**. Longer clips are cropped, shorter ones padded.
- Padding is **circular padding**, not zero padding, and is **applied to the test set too** ("This padding is
  applied to all audio, which means that the test set is also affected").
- **Splicing**: short samples of the same class are concatenated to add training samples. "Each category is assigned the same
  number of spliced samples" -> the same count per class, reducing imbalance.
- Traditional augmentation (training only): added noise (white + environmental, SNR 1-10), time stretching, pitch shifting.

Assumptions (not specified in the paper -- all exposed as config arguments):
- **No environmental noise, white noise only** (simplest form).
- **Number of splices**: the paper only says "the same number" -> `splice_per_class` default 500.
- **Splice candidates**: no criterion for "shorter samples" -> interpreted as cycles shorter than the target length (8 s).
- **Augmentation probability**: not specified -> each waveform augmentation is applied independently with `aug_prob` (default 0.5).
- time stretch 0.9-1.1, pitch shift +/-2 semitones (ranges not specified).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rsc_toolkit.augmentations.domain_transfer import (
    add_white_noise,
    circular_pad,
    pitch_shift,
    sample_snr_db,
    splice_same_class,
    time_stretch,
)
from rsc_toolkit.datasets.icbhi.processing import load_audio, slice_audio
from rsc_toolkit.datasets.index_dataset import IndexDataset
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

logger = logging.getLogger(__name__)


class DTIndexDataset(IndexDataset):
    """
    IndexDataset + (8 kHz cycle -> circular padding to 8 s) + waveform augmentation during training + splice samples.

    Splice samples are kept in the `extra` list and `__getitem__` returns them at indices after the base dataset
    (the original index CSV is left untouched -- raw data stays immutable).
    """

    def __init__(self, *args, target_len: int = 64000, is_train: bool = False,
                 aug_prob: float = 0.5, use_noise: bool = True, use_stretch: bool = True,
                 use_pitch: bool = True, snr_low: float = 1.0, snr_high: float = 10.0,
                 stretch_range=(0.9, 1.1), pitch_semitones: float = 2.0, stretch_impl: str = "linear", **kwargs):
        super().__init__(*args, **kwargs)
        self.target_len = int(target_len)
        self.is_train = bool(is_train)
        self.aug_prob = float(aug_prob)
        self.use_noise, self.use_stretch, self.use_pitch = use_noise, use_stretch, use_pitch
        self.snr_low, self.snr_high = float(snr_low), float(snr_high)
        self.stretch_range = tuple(stretch_range)
        self.pitch_semitones = float(pitch_semitones)
        self.stretch_impl = str(stretch_impl)   # linear (default) | phase_vocoder (variant)
        self.extra: List[Dict[str, Any]] = []   # extra training samples built by splicing (label, list of source rows)

    # ------------------------------------------------------------ load a base cycle
    def _load_cycle(self, row) -> torch.Tensor:
        wav_path = self._resolve_recording_path(str(row["recording_path"]))
        wav = load_audio(wav_path, target_sampling_rate=self.sampling_rate, fade=self.fade)
        return slice_audio(wav, t0=float(row["t0"]), t1=float(row["t1"]), sampling_rate=self.sampling_rate)

    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        """Waveform augmentation during training -- the paper's traditional augmentation (Sec. 2.1), without environmental noise."""
        if self.use_stretch and random.random() < self.aug_prob:
            audio = time_stretch(audio, random.uniform(*self.stretch_range), impl=self.stretch_impl)
        if self.use_pitch and random.random() < self.aug_prob:
            audio = pitch_shift(audio, random.uniform(-self.pitch_semitones, self.pitch_semitones), self.sampling_rate, impl=self.stretch_impl)
        if self.use_noise and random.random() < self.aug_prob:
            audio = add_white_noise(audio, sample_snr_db(self.snr_low, self.snr_high))
        return audio

    # ------------------------------------------------------------ splice preparation
    def build_splices(self, per_class: int, seed: int = 0) -> None:
        """Build `per_class` splice samples per class (concatenating short cycles only)."""
        rng = random.Random(seed)
        target_sec = self.target_len / self.sampling_rate
        short_by_class: Dict[int, List[int]] = {}
        for i, row in self.df.reset_index(drop=True).iterrows():
            lab = row["label"]
            if lab != lab:  # NaN
                continue
            if float(row["t1"]) - float(row["t0"]) < target_sec:
                short_by_class.setdefault(int(lab), []).append(int(i))
        self.extra = []
        for lab, idxs in sorted(short_by_class.items()):
            if not idxs:
                continue
            for _ in range(per_class):
                k = rng.randint(2, 4)  # number of cycles to concatenate (not specified -> 2-4 assumed)
                self.extra.append({"label": lab, "rows": [rng.choice(idxs) for _ in range(k)]})
        logger.info("Built splice samples: %d per class, %d total (added to %d base samples)", per_class, len(self.extra), len(self.df))

    def __len__(self) -> int:
        return len(self.df) + len(self.extra)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n = len(self.df)
        if idx < n:
            row = self.df.iloc[int(idx)]
            audio = self._load_cycle(row)
            if self.is_train:
                audio = self._augment(audio)
            audio = circular_pad(audio, self.target_len)   # applied to test as well (stated in the paper)
            label = row["label"]
            return {
                "audio": audio,
                "label": None if (label != label) else int(label),
                "sample_id": str(row["sample_id"]),
                "recording_path": str(row["recording_path"]),
                "sample_level": "cycle",
                "t0": float(row["t0"]),
                "t1": float(row["t1"]),
            }
        # splice sample
        item = self.extra[idx - n]
        clips = []
        for r in item["rows"]:
            a = self._load_cycle(self.df.iloc[r])
            clips.append(self._augment(a) if self.is_train else a)
        audio = splice_same_class(clips, self.target_len)
        return {
            "audio": audio, "label": int(item["label"]), "sample_id": f"splice_{idx - n}",
            "recording_path": "", "sample_level": "cycle", "t0": 0.0, "t1": 0.0,
        }


class DomainTransferDataModule(IndexDataModule):
    """IndexDataModule + DT dataset replacement + splice generation + balanced class weight."""

    def __init__(self, *args, target_len: int = 64000, splice_per_class: int = 500,
                 aug_prob: float = 0.5, use_noise: bool = True, use_stretch: bool = True,
                 use_pitch: bool = True, stretch_impl: str = "linear", **kwargs):
        self.target_len = int(target_len)
        self.splice_per_class = int(splice_per_class)
        self.aug_kwargs = dict(aug_prob=aug_prob, use_noise=use_noise, use_stretch=use_stretch, use_pitch=use_pitch,
                               stretch_impl=stretch_impl)
        self.train_class_weight: Optional[list] = None
        super().__init__(*args, **kwargs)
        # DTIndexDataset is used at construction time: its specific arguments go through the `_index_dataset_kwargs` hook,
        # and splices are built right after the training dataset is created.
        # (snr 1-10 dB, stretch 0.9-1.1 and pitch +/-2 semitones are DTIndexDataset defaults.)
        if not (isinstance(self.index_dataset_cls, type) and issubclass(self.index_dataset_cls, DTIndexDataset)):
            self.index_dataset_cls = DTIndexDataset

    def _index_dataset_kwargs(self, *, split: str, role: Optional[str] = None):
        kwargs = super()._index_dataset_kwargs(split=split, role=role)
        kwargs.update(target_len=self.target_len, is_train=(role == "train"), **self.aug_kwargs)
        return kwargs

    def _build_index_dataset_single(self, *, split: str, role: Optional[str] = None):
        ds = super()._build_index_dataset_single(split=split, role=role)
        if role == "train" and self.splice_per_class > 0:
            ds.build_splices(self.splice_per_class, seed=0)
        return ds

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self._compute_class_weight()

    def _compute_class_weight(self) -> None:
        """Paper Sec. 2.1 'higher weight for minority classes' -- sklearn balanced (distribution after adding splices)."""
        ds = getattr(self, "train_ds", None)
        base = getattr(ds, "index_ds", ds)
        if base is None:
            return
        from sklearn.utils.class_weight import compute_class_weight

        y = list(base.df["label"].dropna().astype(int)) + [int(e["label"]) for e in getattr(base, "extra", [])]
        y = np.asarray(y)
        classes = np.sort(np.unique(y))
        self.train_class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y).tolist()
        logger.info("Domain Transfer class weight (splices included): %s", [round(w, 4) for w in self.train_class_weight])
