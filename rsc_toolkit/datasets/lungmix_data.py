"""
Lungmix (ICASSP 2025) data path -- **re-implementation from the paper description** (the official repository has no data or preprocessing code).

Items the paper (Sec. III-A/B) **only lists**: 16 kHz resampling, 50-1500 Hz band-pass, 9 s pad/cut, fbank 128x1024. The paper **does not state their order or
where mixing happens** -> the order below is assumed: resample -> band-pass -> 9 s pad/cut (+ original segment length n_valid attached) -> (mixing: online, inside the model's training_forward;
default raw = mix the original segments of length n_valid, then 9 s) -> fbank. Only the mixing position is grounded in the notation of the paper's Eq. 3 (see below).
SPRSound's 7 event classes are unified into the ICBHI-style 4 classes (Fine/Coarse Crackle -> crackle, Both -> both; Normal and Wheeze unchanged).
**Rhonchi and Stridor are excluded from the 4-class view** -- the paper (Sec. III-A) describes merging them into wheeze, but the toolkit does not follow that.
Both classes have 0 events in the 2022 inter-patient test, so test numbers are unchanged; only 54 train events (Rhonchi 39 + Stridor 15) are dropped.

Assumptions (not confirmed by the authors; all exposed as config flags):
- 9 s padding method not specified -> pad_type repeat (toolkit AudioPreprocessor); band-pass is applied to the original segment **before padding** (bandpass.before_pad).
- Mixing position: the maxlen(x_i,x_j) notation in the paper's Eq. 3 and the length-matching structure of the official function suggest 'mix the original segments, then 9 s' -> the default/main run
  uses model mix_stage='raw': the model uses the original segment length (n_valid, sample count before padding; train only, in the meta_labels slot) that this dataset
  puts into the batch with lungmix.emit_valid_len=true (default), mixes only the original segments and then fits them to 9 s. The alternative mix_stage='padded' mixes padded waveforms.
- Band-pass type and order not specified -> 4th-order zero-phase Butterworth (processors/bandpass.py). The application **unit** is not specified either (the paper's
  'band-pass after resampling' in Sec. III-B may refer to whole recordings) -> here sosfiltfilt is applied **per cycle segment** read with resample_after_slice (a zero-phase
  50 Hz high-pass on 1-3 s segments may show edge transients; the per-recording alternative is not implemented).
- SPRSound test not specified as inter/intra -> the 2022 inter-patient test is assumed (toolkit-bundled official_split.txt 'valid' -> 'test').
  The 7->4 mapping is a **derived label** (only the df label column is replaced; the original index and wav files are unchanged).

Data flow: LungmixDataModule (IndexDataModule subclass, data_registry 'lungmix') -> passes bandpass/label_map arguments to LungmixIndexDataset (index_dataset_cls).
features.kind: waveform, so no cache (training mixes online, which rules out caching; evaluation also goes waveform -> in-model fbank).
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

# Toolkit SPRSound event index (.datasets/sprsound/index/index_cycle.csv, LabelEncoder first-appearance order) -- label id <-> name.
# Verified: the event label counts of the bundled metadata.csv (Normal 6199 / Fine Crackle 992 / Wheeze 757 / Coarse Crackle 52 / Both 31 /
# Rhonchi 39 / Stridor 15, all distinct) match the per-id counts of the index (tests/unit/models/test_ast_lungmix.py).
SPRSOUND_EVENT_ID_TO_NAME = {0: "Normal", 1: "Fine Crackle", 2: "Wheeze", 3: "Coarse Crackle", 4: "Both", 5: "Rhonchi", 6: "Stridor"}
# ICBHI 4-class view (normal 0 / crackle 1 / wheeze 2 / both 3). Rhonchi and Stridor have no ICBHI counterpart and are **excluded**
# (not merged into wheeze; this differs from the 'merged into wheeze' description in LungMix Sec. III-A, QLung Sec. 3.1 and BTS-CARD Sec. 3.1,
#  and matches the removal in the official BTS-CARD notebook). apply_label_map filters rows whose label is not in the map out of the dataframe.
SPRSOUND_4CLASS_EXCLUDED = ("Rhonchi", "Stridor")
SPRSOUND_NAME_TO_4CLASS = {"Normal": 0, "Fine Crackle": 1, "Coarse Crackle": 1, "Wheeze": 2, "Both": 3}
SPRSOUND_7_TO_4 = {i: SPRSOUND_NAME_TO_4CLASS[n] for i, n in SPRSOUND_EVENT_ID_TO_NAME.items() if n in SPRSOUND_NAME_TO_4CLASS}
LABEL_MAPS = {"none": None, "sprsound_7to4": SPRSOUND_7_TO_4}


def map_labels(labels, label_map: Optional[Dict[int, int]]):
    """Replace the label column using the map. Labels not in the map (excluded classes) become None -- use apply_label_map to drop rows."""
    if label_map is None:
        return labels
    return [label_map.get(int(v)) if not pd.isna(v) else v for v in labels]


def apply_label_map(df: pd.DataFrame, label_map: Dict[int, int], *, where: str = "") -> tuple[pd.DataFrame, int]:
    """
    Apply a derived label map to an index dataframe: **drop** rows whose label is not in the map (reset_index) and replace the rest.
    The original label is kept in the label_original column. Returns (new df, number of dropped rows). The input df is not modified.
    """
    labels = pd.to_numeric(df["label"], errors="coerce")
    keep = labels.isna() | labels.isin(list(label_map.keys()))
    dropped = int((~keep).sum())
    out = df.loc[keep].reset_index(drop=True).copy()
    out["label_original"] = out["label"]
    out["label"] = map_labels(out["label"].tolist(), label_map)
    if dropped:
        excluded = sorted({int(v) for v in labels[~keep].dropna().unique()})
        logger.info("apply_label_map%s: labels %s not in the map, dropped %d rows -> %d rows", f"({where})" if where else "", excluded, dropped, len(out))
    return out, dropped


class _BandpassThenPreprocess:
    """
    Wrapper plugged into IndexDataset.audio_preprocessor: (optional) band-pass (before padding) -> original preprocessing (9 s pad/cut).
    Each call stores the original segment length before padding in last_raw_len (LungmixIndexDataset puts it into the batch as n_valid; safe because calls are sequential per worker).
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
    """IndexDataset + (optional) band-pass + (optional) derived label mapping. The original index/files are only read."""

    N_CLS = 4   # fixed 4 classes as in the paper (ClassLayout normal/crackle/wheeze/both)

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
            raise ValueError(f"label_map must be one of {sorted(LABEL_MAPS)}: {label_map!r}")
        self.label_map = LABEL_MAPS[self.label_map_name]
        labels = pd.to_numeric(self.df["label"], errors="coerce").dropna()
        if self.label_map is None and len(labels) and int(labels.max()) >= self.N_CLS:
            raise ValueError(
                f"label_map='none' is for 4-class indexes (ICBHI) only, but the maximum label is {int(labels.max())} -- for SPRSound 7-class "
                "use lungmix.label_map=sprsound_7to4 (the toolkit's standard 7-class setup is outside this data path)")
        if self.label_map is not None:
            self.df, dropped = apply_label_map(self.df, self.label_map, where=f"LungmixIndexDataset split={self.split}")
            counts = self.df["label"].value_counts().sort_index().to_dict()
            logger.info("LungmixIndexDataset(split=%s): applied label map %s -> %s (dropped %d rows)", self.split, self.label_map_name, counts, dropped)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        raw = self._pre.last_raw_len
        out["n_valid"] = int(min(raw, out["audio"].shape[-1])) if raw is not None else int(out["audio"].shape[-1])
        return out


def valid_len_meta(item: Dict[str, Any]) -> int:
    """meta_label_fn: puts the original segment length (sample count) into the meta_labels slot (mix_stage='raw')."""
    return int(item["n_valid"])


class LungmixDataModule(IndexDataModule):
    """IndexDataModule + passes the lungmix block (bandpass, label_map) to LungmixIndexDataset."""

    meta_kind = "lungmix_valid_len"  # train batch[2] = original segment length (emit_valid_len) -- for ast_lungmix mix_stage=raw

    def __init__(self, *args, lungmix: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(lungmix or {})
        self.lungmix_bandpass = dict(cfg.get("bandpass") or {})
        self.lungmix_label_map = str(cfg.get("label_map", "none"))
        self.emit_valid_len = bool(cfg.get("emit_valid_len", True))    # train batch meta_labels = original segment length (default mix_stage raw path)

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None, split_role=None):
        if split_role is None:  # direct call (outside setup), backward compatible -- identity comparison
            split_role = "train" if base_ds is getattr(self, "train_ds", None) else "val"
        if self.emit_valid_len and meta_label_fn is None and split_role == "train":
            meta_label_fn = valid_len_meta
        return super()._wrap_with_features(base_ds, feature_transform=feature_transform, meta_label_fn=meta_label_fn,
                                           split_role=split_role)

    def _index_dataset_kwargs(self, *, split: str, role: Optional[str] = None) -> Dict[str, Any]:
        # Add only the lungmix-specific arguments to the parent's assembly (including the fade/resample_after_slice/time_unit/split_file rules)
        kwargs = super()._index_dataset_kwargs(split=split, role=role)
        kwargs.update(bandpass=self.lungmix_bandpass, label_map=self.lungmix_label_map)
        return kwargs


def sprsound_4class_counts(metadata_csv) -> Dict[str, Dict[int, int]]:
    """Event labels per split of the bundled metadata.csv -> 4-class counts (for tests/docs). Excluded classes (Rhonchi, Stridor) are not counted."""
    df = pd.read_csv(metadata_csv)
    out: Dict[str, Dict[int, int]] = {}
    for split, sub in df.groupby("Split"):
        c = sub["Label"].map(SPRSOUND_NAME_TO_4CLASS).value_counts().sort_index()
        out[str(split)] = {int(k): int(v) for k, v in c.items()}
    return out
