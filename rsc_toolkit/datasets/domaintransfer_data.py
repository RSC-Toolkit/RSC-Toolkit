"""
Domain Transfer(Wang & Wang, ICASSP 2022) 의 ICBHI 데이터 경로 — **논문 명세 재구현** (코드 없음, 등급 C).

논문 §2.1 · §3.2 가 명시한 것:
- 전 오디오를 **8 kHz 로 리샘플**, **약 8 초**로 길이 통일. 긴 것은 자르고 짧은 것은 채운다.
- 채우는 방식은 zero padding 이 아니라 **circular padding** 이며 **test set 에도 적용**한다("This padding is
  applied to all audio, which means that the test set is also affected").
- **Splicing**: 같은 클래스의 짧은 샘플들을 이어붙여 학습 샘플을 늘린다. "Each category is assigned the same
  number of spliced samples" → 클래스마다 같은 개수를 만들어 불균형을 줄인다.
- 전통 증강(학습만): 잡음 추가(백색 + 환경, SNR 1–10), time stretching, pitch shifting.

가정(논문 미기재 — 전부 config 인자, AUTHOR_QUESTIONS 참조):
- **환경잡음 미적용, 백색잡음만**(사용자 결정 2026-08-30 "가장 단순한 형태로"). 표에 병기한다.
- **splice 개수**: "같은 개수" 라고만 하고 몇 개인지 없다 → `splice_per_class` 기본 500.
- **splice 대상**: "shorter samples" 의 기준이 없다 → 목표 길이(8 s) 미만인 사이클로 해석.
- **증강 적용 확률**: 미기재 → 각 파형 증강을 `aug_prob` (기본 0.5) 로 독립 적용.
- time stretch 0.9–1.1 · pitch shift ±2 semitone (범위 미기재).
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
    IndexDataset + (8 kHz 사이클 → circular padding 8 s) + 학습 시 파형 증강 + splice 샘플.

    splice 샘플은 `extra` 목록으로 들고 있다가 `__getitem__` 이 원 데이터셋 뒤 인덱스로 돌려준다
    (원본 인덱스 CSV 는 건드리지 않는다 — 원본 데이터 불변 원칙).
    """

    def __init__(self, *args, target_len: int = 64000, is_train: bool = False,
                 aug_prob: float = 0.5, use_noise: bool = True, use_stretch: bool = True,
                 use_pitch: bool = True, snr_low: float = 1.0, snr_high: float = 10.0,
                 stretch_range=(0.9, 1.1), pitch_semitones: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_len = int(target_len)
        self.is_train = bool(is_train)
        self.aug_prob = float(aug_prob)
        self.use_noise, self.use_stretch, self.use_pitch = use_noise, use_stretch, use_pitch
        self.snr_low, self.snr_high = float(snr_low), float(snr_high)
        self.stretch_range = tuple(stretch_range)
        self.pitch_semitones = float(pitch_semitones)
        self.extra: List[Dict[str, Any]] = []   # splice 로 만든 추가 학습 샘플 (label, source row 목록)

    # ------------------------------------------------------------ 원 사이클 로드
    def _load_cycle(self, row) -> torch.Tensor:
        wav_path = self._resolve_recording_path(str(row["recording_path"]))
        wav = load_audio(wav_path, target_sampling_rate=self.sampling_rate, fade=self.fade)
        return slice_audio(wav, t0=float(row["t0"]), t1=float(row["t1"]), sampling_rate=self.sampling_rate)

    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        """학습 시 파형 증강 — 논문 §2.1 의 traditional augmentation (환경잡음은 미적용)."""
        if self.use_stretch and random.random() < self.aug_prob:
            audio = time_stretch(audio, random.uniform(*self.stretch_range))
        if self.use_pitch and random.random() < self.aug_prob:
            audio = pitch_shift(audio, random.uniform(-self.pitch_semitones, self.pitch_semitones), self.sampling_rate)
        if self.use_noise and random.random() < self.aug_prob:
            audio = add_white_noise(audio, sample_snr_db(self.snr_low, self.snr_high))
        return audio

    # ------------------------------------------------------------ splice 준비
    def build_splices(self, per_class: int, seed: int = 0) -> None:
        """클래스마다 `per_class` 개의 splice 샘플을 만든다(짧은 사이클만 이어붙임)."""
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
                k = rng.randint(2, 4)  # 몇 개를 이어붙일지 (미기재 → 2–4 가정)
                self.extra.append({"label": lab, "rows": [rng.choice(idxs) for _ in range(k)]})
        logger.info("splice 샘플 생성: 클래스당 %d개, 총 %d개 (원 %d개에 추가)", per_class, len(self.extra), len(self.df))

    def __len__(self) -> int:
        return len(self.df) + len(self.extra)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n = len(self.df)
        if idx < n:
            row = self.df.iloc[int(idx)]
            audio = self._load_cycle(row)
            if self.is_train:
                audio = self._augment(audio)
            audio = circular_pad(audio, self.target_len)   # test 에도 동일 적용(논문 명시)
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
        # splice 샘플
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
    """IndexDataModule + DT 데이터셋 교체 + splice 생성 + balanced class weight."""

    def __init__(self, *args, target_len: int = 64000, splice_per_class: int = 500,
                 aug_prob: float = 0.5, use_noise: bool = True, use_stretch: bool = True,
                 use_pitch: bool = True, **kwargs):
        self.target_len = int(target_len)
        self.splice_per_class = int(splice_per_class)
        self.aug_kwargs = dict(aug_prob=aug_prob, use_noise=use_noise, use_stretch=use_stretch, use_pitch=use_pitch)
        self.train_class_weight: Optional[list] = None
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        for attr, is_train in (("train_ds", True), ("val_ds", False), ("test_ds", False)):
            ds = getattr(self, attr, None)
            base = getattr(ds, "index_ds", ds)
            if base is None or isinstance(base, DTIndexDataset):
                continue
            new = DTIndexDataset.__new__(DTIndexDataset)
            new.__dict__.update(base.__dict__)
            new.target_len, new.is_train = self.target_len, is_train
            new.snr_low, new.snr_high = 1.0, 10.0
            new.stretch_range, new.pitch_semitones = (0.9, 1.1), 2.0
            new.extra = []
            for k, v in self.aug_kwargs.items():
                setattr(new, k, v)
            if is_train and self.splice_per_class > 0:
                new.build_splices(self.splice_per_class, seed=0)
            if ds is base:
                setattr(self, attr, new)
            else:
                ds.index_ds = new
        self._compute_class_weight()

    def _compute_class_weight(self) -> None:
        """논문 §2.1 '소수 클래스 가중치 ↑' — sklearn balanced (splice 포함 후 분포 기준)."""
        ds = getattr(self, "train_ds", None)
        base = getattr(ds, "index_ds", ds)
        if base is None:
            return
        from sklearn.utils.class_weight import compute_class_weight

        y = list(base.df["label"].dropna().astype(int)) + [int(e["label"]) for e in getattr(base, "extra", [])]
        y = np.asarray(y)
        classes = np.sort(np.unique(y))
        self.train_class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y).tolist()
        logger.info("Domain Transfer class weight(splice 포함): %s", [round(w, 4) for w in self.train_class_weight])
