"""
Lungmix(ICASSP 2025) 데이터 경로 — **논문 명세 재구현** (공식 저장소에 데이터·전처리 코드 없음).

논문 §III-A/B 가 **나열만** 한 항목: 16 kHz 리샘플, 밴드패스 50–1500 Hz, 9 s pad/cut, fbank 128×1024. 논문은 이들의 **순서와 믹싱 위치를
적지 않았다** → 아래 순서는 가정: 리샘플 → 밴드패스 → 9 s pad/cut(+원 구간 길이 n_valid 동봉) → (믹싱: 모델 training_forward 안, 온라인;
기본 raw = n_valid 만큼의 원 구간을 섞은 뒤 9 s) → fbank. 믹싱 위치만은 논문 식 3 표기에 근거(아래 #5).
SPRSound 는 이벤트 7-class 를 ICBHI 식 4-class 로 통일(Fine/Coarse Crackle→crackle, Stridor/Rhonchi→wheeze, Both→both).

가정(저자 확인 전, 전부 config 플래그 — docs/AUTHOR_QUESTIONS_2026-08-26.md LungMix 절):
- #5 9 s 패딩 방식 미기재 → pad_type repeat(툴킷 AudioPreprocessor), 밴드패스는 **패딩 전** 원 구간에 적용(bandpass.before_pad).
- #5 믹싱 위치: 논문 식 3 의 maxlen(x_i,x_j) 표기와 공식 함수의 길이 맞춤 구조는 '원 구간을 섞은 뒤 9 s' 를 시사 → 기본·주 런은
  model mix_stage='raw': 이 데이터셋이 lungmix.emit_valid_len=true(기본) 로 배치에 실어 주는 원 구간 길이(n_valid, 패딩 전 샘플 수;
  train 만, meta_labels 자리) 를 모델이 써서 원 구간만 섞은 뒤 9 s 로 맞춘다. 대안 mix_stage='padded' 는 패딩된 파형을 섞는다.
- #6 밴드패스 종류·차수 미기재 → Butterworth 4차 zero-phase (processors/bandpass.py). 적용 **단위**도 미기재(논문 §III-B '리샘플 후
  밴드패스' 는 녹음 전체 기준일 수 있음) → 여기서는 resample_after_slice 로 읽은 **사이클 구간 단위** sosfiltfilt (1–3 s 구간에
  50 Hz 하이패스 zero-phase 라 양끝 과도응답 가능; 녹음 단위 대안은 미구현).
- #7 SPRSound test 가 inter/intra 미기재 → 2022 inter-patient test(툴킷 번들 official_split.txt 'valid' → 'test') 가정.
  7→4 매핑은 **파생 라벨**(df 의 label 열만 치환; 원본 index·wav 는 불변).

데이터 흐름: LungmixDataModule(IndexDataModule 서브클래스, data_registry 'lungmix') → LungmixIndexDataset(index_dataset_cls) 에
bandpass/label_map 인자를 넘긴다. features.kind: waveform 이라 캐시 없음(학습은 온라인 믹싱이라 캐시 불가, 평가도 파형→모델 내부 fbank).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from rsc_toolkit.datasets.index_dataset import IndexDataset
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

logger = logging.getLogger(__name__)

# 툴킷 SPRSound 이벤트 인덱스(.datasets/sprsound/index/index_cycle.csv, LabelEncoder 첫 등장 순) — 라벨 id ↔ 이름.
# 검증: 번들 metadata.csv 의 이벤트 라벨 개수(Normal 6199 / Fine Crackle 992 / Wheeze 757 / Coarse Crackle 52 / Both 31 /
# Rhonchi 39 / Stridor 15, 전부 서로 다름)가 인덱스 id 별 개수와 일치 (tests/unit/models/test_ast_lungmix.py).
SPRSOUND_EVENT_ID_TO_NAME = {0: "Normal", 1: "Fine Crackle", 2: "Wheeze", 3: "Coarse Crackle", 4: "Both", 5: "Rhonchi", 6: "Stridor"}
# 논문 §III-A 4-class 통일 (ICBHI 순서 normal 0 / crackle 1 / wheeze 2 / both 3)
SPRSOUND_NAME_TO_4CLASS = {
    "Normal": 0, "Fine Crackle": 1, "Coarse Crackle": 1, "Wheeze": 2, "Rhonchi": 2, "Stridor": 2, "Both": 3,
}
SPRSOUND_7_TO_4 = {i: SPRSOUND_NAME_TO_4CLASS[n] for i, n in SPRSOUND_EVENT_ID_TO_NAME.items()}
LABEL_MAPS = {"none": None, "sprsound_7to4": SPRSOUND_7_TO_4}


def map_labels(labels, label_map: Optional[Dict[int, int]]):
    if label_map is None:
        return labels
    return [label_map[int(v)] if not pd.isna(v) else v for v in labels]


class _BandpassThenPreprocess:
    """
    IndexDataset.audio_preprocessor 자리에 끼우는 래퍼: (선택) 밴드패스(패딩 전) → 원 전처리(9 s pad/cut).
    호출마다 패딩 전 원 구간 길이를 last_raw_len 에 남긴다(LungmixIndexDataset 이 n_valid 로 배치에 싣는다; 워커마다 순차 호출이라 안전).
    """

    def __init__(self, bandpass, inner, before_pad: bool = True):
        self.bandpass, self.inner, self.before_pad = bandpass, inner, bool(before_pad)
        self.last_raw_len: Optional[int] = None

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        self.last_raw_len = int(audio.shape[-1])
        if self.bandpass is None:
            return self.inner(audio) if self.inner is not None else audio
        if self.before_pad:
            audio = self.bandpass(audio)
            return self.inner(audio) if self.inner is not None else audio
        audio = self.inner(audio) if self.inner is not None else audio
        return self.bandpass(audio)


class LungmixIndexDataset(IndexDataset):
    """IndexDataset + (선택) 밴드패스 + (선택) 파생 라벨 매핑. 원본 인덱스/파일은 읽기만 한다."""

    N_CLS = 4   # 논문 4-class 고정 (ClassLayout normal/crackle/wheeze/both)

    def __init__(self, *args, bandpass: Optional[dict] = None, label_map: Optional[str] = "none", **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(bandpass or {})
        self.bandpass = None
        if cfg.get("enabled", False):
            from rsc_toolkit.processors.bandpass import BandpassFilter

            self.bandpass = BandpassFilter(
                sampling_rate=self.sampling_rate, low_hz=cfg.get("low_hz", 50.0), high_hz=cfg.get("high_hz", 1500.0),
                order=cfg.get("order", 4), zero_phase=cfg.get("zero_phase", True), kind=cfg.get("kind", "butter"),
            )
        self._pre = _BandpassThenPreprocess(self.bandpass, self.audio_preprocessor, before_pad=cfg.get("before_pad", True))
        self.audio_preprocessor = self._pre
        self.label_map_name = str(label_map or "none")
        if self.label_map_name not in LABEL_MAPS:
            raise ValueError(f"label_map 은 {sorted(LABEL_MAPS)} 중 하나여야 한다: {label_map!r}")
        self.label_map = LABEL_MAPS[self.label_map_name]
        labels = pd.to_numeric(self.df["label"], errors="coerce").dropna()
        if self.label_map is None and len(labels) and int(labels.max()) >= self.N_CLS:
            raise ValueError(
                f"label_map='none' 은 4-class 인덱스(ICBHI) 전용인데 라벨 최대값 {int(labels.max())} 이 있다 — SPRSound 7-class 는 "
                "lungmix.label_map=sprsound_7to4 를 쓰라 (툴킷 표준 7-class 는 이 데이터 경로 밖)")
        if self.label_map is not None:
            self.df = self.df.copy()
            self.df["label_original"] = self.df["label"]
            self.df["label"] = map_labels(self.df["label"].tolist(), self.label_map)
            counts = self.df["label"].value_counts().sort_index().to_dict()
            logger.info("LungmixIndexDataset(split=%s): 라벨 매핑 %s 적용 → %s", self.split, self.label_map_name, counts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        raw = self._pre.last_raw_len
        out["n_valid"] = int(min(raw, out["audio"].shape[-1])) if raw is not None else int(out["audio"].shape[-1])
        return out


def valid_len_meta(item: Dict[str, Any]) -> int:
    """meta_label_fn: 원 구간 길이(샘플 수) 를 meta_labels 자리에 싣는다 (mix_stage='raw')."""
    return int(item["n_valid"])


class LungmixDataModule(IndexDataModule):
    """IndexDataModule + lungmix 블록(bandpass·label_map) 을 LungmixIndexDataset 에 전달."""

    meta_kind = "lungmix_valid_len"  # 학습 batch[2] = 원 구간 길이 (emit_valid_len) — ast_lungmix mix_stage=raw 용

    def __init__(self, *args, lungmix: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(lungmix or {})
        self.lungmix_bandpass = dict(cfg.get("bandpass") or {})
        self.lungmix_label_map = str(cfg.get("label_map", "none"))
        self.emit_valid_len = bool(cfg.get("emit_valid_len", True))    # train 배치 meta_labels = 원 구간 길이 (mix_stage raw 기본 경로)

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None, split_role=None):
        if split_role is None:  # 직접 호출(setup 밖) 하위호환 — 예전 identity 비교
            split_role = "train" if base_ds is getattr(self, "train_ds", None) else "val"
        if self.emit_valid_len and meta_label_fn is None and split_role == "train":
            meta_label_fn = valid_len_meta
        return super()._wrap_with_features(base_ds, feature_transform=feature_transform, meta_label_fn=meta_label_fn,
                                           split_role=split_role)

    def _index_dataset_kwargs(self, *, split: str) -> Dict[str, Any]:
        # 부모의 조립(fade/resample_after_slice/time_unit/split_file 규칙 포함)에 lungmix 전용 인자만 더한다
        kwargs = super()._index_dataset_kwargs(split=split)
        kwargs.update(bandpass=self.lungmix_bandpass, label_map=self.lungmix_label_map)
        return kwargs


def sprsound_4class_counts(metadata_csv) -> Dict[str, Dict[int, int]]:
    """번들 metadata.csv 의 split 별 이벤트 라벨 → 4-class 개수 (테스트·문서용)."""
    df = pd.read_csv(metadata_csv)
    out: Dict[str, Dict[int, int]] = {}
    for split, sub in df.groupby("Split"):
        c = sub["Label"].map(SPRSOUND_NAME_TO_4CLASS).value_counts().sort_index()
        out[str(split)] = {int(k): int(v) for k, v in c.items()}
    return out
