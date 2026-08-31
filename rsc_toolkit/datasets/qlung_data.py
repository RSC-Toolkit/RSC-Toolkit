"""
QLung 데이터 경로 — AQS(음향 품질 점수) 를 sample_id 별로 캐시해 배치의 3번째 원소(meta_labels 자리, **float**) 로 실어 보낸다.

공식 util/icbhi_dataset.py:170-177: 데이터셋 생성 시 사이클마다 (8 s repeat 패딩 + fade_out 된 16 kHz 파형에서)
  rms_energy = compute_rms_energy(wave), spectral_entropy = compute_spectral_entropy(wave),
  quality_score = compute_quality_score_entropy_rms(rms_energy, rms_energy, α, β)   ← 엔트로피 자리에 rms (버그, 기본값 유지)
를 계산해 __getitem__ 이 (image, label, rms, entropy, quality) 를 돌려준다. train 루프는 quality_score 만 쓴다.

툴킷 구현:
  QLungAQSIndexDataset(IndexDataset): 항목에 out["aqs"] 추가. rms/entropy 는 파형에서 torch 로 계산(methods/qlung.py) 하고
    downloaded_datasets/aqs_cache/<dataset>_acoustic_<전처리키>.pt (sample_id → {rms, entropy}) 에 캐시(변형 무관, 1회).
    변형별 AQS 도 <dataset>_<variant>.pt 로 남긴다(집계·검사용; 런타임은 rms/entropy 에서 순수 함수로 계산).
  QLungDataModule(IndexDataModule, data_registry 'qlung'): qlung 블록 {aqs_alpha, aqs_beta, entropy_source, formula, cache_dir,
    precompute, attach_val} 을 데이터셋에 넘기고, setup 에서 누락 sample_id 의 rms/entropy 를 DataLoader(num_workers) 로 일괄 계산.
    IndexFeatureDataset 대신 _QLungFeatureDataset 이 (feats, label, sample_id, aqs float) 를 내고 collate 가 float64 텐서로 만든다(공식 default_collate 와 동일).
    val 배치에도 AQS 를 붙인다(attach_val 기본 true; 원본 val loader 도 quality 를 돌려주되 미사용 — 모델 eval 경로는 무시).
  전처리키 = (sampling_rate, desired_length_sec, pad_type, fade, resample_after_slice, time_unit) — 파형이 다르면 캐시가 갈린다.
SPRSound 도 같은 클래스로 동작한다(OOD 평가엔 AQS 가 필요 없지만 형식을 맞춘다; label_map 으로 7→4 파생 라벨).
"""
from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rsc_toolkit.datasets.index_dataset import IndexDataset, IndexFeatureDataset
from rsc_toolkit.methods.qlung import AQS_ENTROPY_SOURCES, aqs_from_features, aqs_variant_name, compute_acoustic_features
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

logger = logging.getLogger(__name__)

DEFAULT_AQS_CACHE_DIR = "downloaded_datasets/aqs_cache"


class AQSConfig:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, entropy_source: str = "rms", formula: str = "code",
                 cache_dir: str = DEFAULT_AQS_CACHE_DIR, precompute: bool = True, attach_val: bool = True,
                 precompute_num_workers: int = 8, precompute_batch_size: int = 16):
        if entropy_source not in AQS_ENTROPY_SOURCES:
            raise ValueError(f"entropy_source must be one of {AQS_ENTROPY_SOURCES}, got {entropy_source!r}")
        if formula not in ("code", "paper"):
            raise ValueError(f"formula must be code|paper, got {formula!r}")
        self.alpha, self.beta = float(alpha), float(beta)
        self.entropy_source, self.formula = entropy_source, formula
        self.cache_dir = cache_dir
        self.precompute = bool(precompute)
        self.attach_val = bool(attach_val)
        self.precompute_num_workers = int(precompute_num_workers)
        self.precompute_batch_size = int(precompute_batch_size)

    @property
    def variant(self) -> str:
        return aqs_variant_name(self.alpha, self.beta, self.entropy_source, self.formula)

    def aqs(self, rms: float, entropy: float, entropy_nanmean: Optional[float] = None) -> float:
        return aqs_from_features(rms, entropy, alpha=self.alpha, beta=self.beta, entropy_source=self.entropy_source,
                                 formula=self.formula, entropy_nanmean=entropy_nanmean)

    def aqs_of(self, feats: Dict[str, float]) -> float:
        return self.aqs(feats["rms"], feats["entropy"], feats.get("entropy_nanmean"))


class AcousticFeatureCache:
    """sample_id → {'rms','entropy'} 의 단일 .pt 파일 캐시 (원자적 저장)."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.data: Dict[str, Dict[str, float]] = {}
        if self.path.exists():
            self.data = torch.load(self.path, map_location="cpu", weights_only=False)
            logger.info("AQS 음향 특징 캐시 로드: %s (%d)", self.path, len(self.data))

    def missing(self, sample_ids: List[str]) -> List[str]:
        return [s for s in sample_ids if s not in self.data]

    def update(self, items: Dict[str, Dict[str, float]]) -> None:
        self.data.update(items)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".pt.tmp.{os.getpid()}")
        torch.save(self.data, tmp)
        os.replace(tmp, self.path)


class QLungAQSIndexDataset(IndexDataset):
    """IndexDataset + out['aqs'] (float). 캐시에 없으면 그 자리에서 계산해 메모리 캐시에 넣는다(파일 저장은 datamodule)."""

    N_CLS = 4

    def __init__(self, *args, aqs: Optional[AQSConfig] = None, acoustic_cache: Optional[AcousticFeatureCache] = None,
                 label_map: Optional[Dict[int, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aqs_cfg = aqs or AQSConfig()
        self.acoustic_cache = acoustic_cache or AcousticFeatureCache(pathlib.Path("/nonexistent/aqs.pt"))
        self.label_map = label_map
        if label_map is not None:
            self.df = self.df.copy()
            self.df["label_original"] = self.df["label"]
            self.df["label"] = [label_map[int(v)] if not pd.isna(v) else v for v in self.df["label"].tolist()]
            counts = self.df["label"].value_counts().sort_index().to_dict()
            logger.info("QLungAQSIndexDataset(split=%s): 라벨 매핑 적용 → %s", self.split, counts)

    def sample_ids(self) -> List[str]:
        return self.df["sample_id"].astype(str).tolist()

    def acoustic_features(self, idx: int) -> Tuple[str, Dict[str, float]]:
        """파형 로드(전처리 포함) → {rms, entropy}. 캐시 미사용(계산 전용)."""
        item = IndexDataset.__getitem__(self, idx)
        return str(item["sample_id"]), compute_acoustic_features(item["audio"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        sid = str(out["sample_id"])
        feats = self.acoustic_cache.data.get(sid)
        if feats is None:
            feats = compute_acoustic_features(out["audio"])
            self.acoustic_cache.data[sid] = feats
        out["rms"], out["entropy"] = feats["rms"], feats["entropy"]
        out["aqs"] = self.aqs_cfg.aqs_of(feats)
        return out


class _QLungFeatureDataset(IndexFeatureDataset):
    """(feats, label, sample_id, aqs float) — IndexFeatureDataset 의 int meta 대신 float."""

    def __init__(self, index_ds, feature_processor, feature_transform=None, attach_aqs: bool = True):
        super().__init__(index_ds, feature_processor, feature_transform=feature_transform, meta_label_fn=None)
        self.attach_aqs = attach_aqs

    def __getitem__(self, i):
        item = self.index_ds[i]
        audio = item["audio"].squeeze()
        feats = self.fp.make_features_cached(audio, sample_id=item["sample_id"])
        if self.feature_transform is not None:
            feats = self.feature_transform(feats)
        if self.attach_aqs:
            return feats, item["label"], item["sample_id"], float(item["aqs"])
        return feats, item["label"], item["sample_id"]


def qlung_collate(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    if len(batch[0]) >= 4:
        aqs = torch.as_tensor([b[3] for b in batch], dtype=torch.float64)  # 공식: python float → default_collate float64 (loss.py:60 부터 float64)
        return x, y, aqs
    return x, y


class _AcousticJob(Dataset):
    def __init__(self, ds: QLungAQSIndexDataset, indices: List[int]):
        self.ds, self.indices = ds, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds.acoustic_features(self.indices[i])


def precompute_acoustic_features(ds: QLungAQSIndexDataset, cache: AcousticFeatureCache, *, num_workers: int = 8,
                                 batch_size: int = 16, split_name: str = "") -> int:
    sids = ds.sample_ids()
    missing = set(cache.missing(sids))
    if not missing:
        logger.info("AQS 음향 특징 캐시 완비 split=%s (%d)", split_name, len(sids))
        return 0
    idx = [i for i, s in enumerate(sids) if s in missing]
    logger.info("AQS 음향 특징 계산 split=%s: %d/%d 누락 (workers=%d)", split_name, len(idx), len(sids), num_workers)
    t0 = time.time()
    loader = DataLoader(_AcousticJob(ds, idx), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=lambda b: b)
    done = 0
    with torch.inference_mode():
        for batch in loader:
            cache.update({sid: f for sid, f in batch})
            done += len(batch)
            if done % 800 < len(batch):
                logger.info("  AQS 특징 %d/%d (%.1fs)", done, len(idx), time.time() - t0)
    cache.save()
    logger.info("AQS 음향 특징 계산 완료 split=%s: %d 개, %.1fs → %s", split_name, done, time.time() - t0, cache.path)
    return done


class QLungDataModule(IndexDataModule):
    """IndexDataModule + qlung 블록(AQS 설정) — 배치 (x, y, aqs float64)."""

    def __init__(self, *args, qlung: Optional[dict] = None, label_map: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(qlung or {})
        self.aqs_cfg = AQSConfig(**cfg)
        cache_dir = self._resolve_path(self.aqs_cfg.cache_dir)
        self.acoustic_cache = AcousticFeatureCache(cache_dir / f"{self.dataset_name}_acoustic_{self.preprocess_key()}.pt")
        self.aqs_variant_path = cache_dir / f"{self.dataset_name}_{self.aqs_cfg.variant}_{self.preprocess_key()}.pt"
        self.label_map_name = label_map
        self.label_map = None
        if label_map == "sprsound_7to4":
            from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4
            self.label_map = dict(SPRSOUND_7_TO_4)
        elif label_map not in (None, "none"):
            raise ValueError(f"label_map 은 none|sprsound_7to4: {label_map!r}")

    def preprocess_key(self) -> str:
        return (f"sr{self.sampling_rate}_len{self.desired_length_sec:g}_{self.pad_type}_fade{int(self.fade)}"
                f"_ras{int(self.resample_after_slice)}_{self.time_unit}")

    def _build_index_dataset_single(self, *, split: str):
        if self.index_csv is None:
            self.index_csv = self._default_index_path()
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index csv not found: {self.index_csv}")
        kwargs: Dict[str, Any] = dict(
            index_csv=self.index_csv, data_root=self.raw_data_directory, sampling_rate=self.sampling_rate,
            sample_level=self.sample_level, split=split, audio_preprocessor=self.audio_preprocessor,
            aqs=self.aqs_cfg, acoustic_cache=self.acoustic_cache, label_map=self.label_map,
        )
        if not self.fade:
            kwargs["fade"] = False
        if self.resample_after_slice:
            kwargs["resample_after_slice"] = True
        if self.time_unit != "s":
            kwargs["time_unit"] = self.time_unit
        if self.split_file is not None:
            kwargs["split_file"] = self.split_file
        return self.index_dataset_cls(**kwargs)

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None):
        from torch.utils.data import ConcatDataset
        from rsc_toolkit.datasets.feature_cache import FeatureCache

        cache_dir = pathlib.Path(self.features_cfg.cache_dir) if self.features_cfg.cache_dir else (self.outputs_directory / "features_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        feature_cache = FeatureCache(cache_dir=cache_dir)
        fp = self._build_feature_processor(feature_cache)
        feature_cache.configure(fp.get_processor_config(), processor_name=fp.__class__.__name__,
                                audio_chain=self._audio_chain())
        attach = True if base_ds is getattr(self, "train_ds", None) else self.aqs_cfg.attach_val
        if isinstance(base_ds, ConcatDataset):
            # 여러 split 결합(SPRSound train+test) — 캐시 웜업이 index_ds.df 를 못 보므로 IndexFeatureDataset 과 같은 fallback 사용
            pass
        return _QLungFeatureDataset(base_ds, fp, feature_transform=feature_transform, attach_aqs=attach)

    def _each_index_ds(self, ds) -> List[QLungAQSIndexDataset]:
        from torch.utils.data import ConcatDataset
        if isinstance(ds, ConcatDataset):
            return [d for sub in ds.datasets for d in self._each_index_ds(sub)]
        return [ds]

    def _precompute_aqs(self, ds, split_name: str) -> None:
        if not self.aqs_cfg.precompute or not self._is_global_zero():
            return
        for sub in self._each_index_ds(ds):
            precompute_acoustic_features(sub, self.acoustic_cache, num_workers=self.aqs_cfg.precompute_num_workers,
                                         batch_size=self.aqs_cfg.precompute_batch_size, split_name=split_name)
        self._write_variant_file()

    def _write_variant_file(self) -> None:
        d = {sid: self.aqs_cfg.aqs_of(f) for sid, f in self.acoustic_cache.data.items()}
        self.aqs_variant_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.aqs_variant_path.with_suffix(f".pt.tmp.{os.getpid()}")
        torch.save({"variant": self.aqs_cfg.variant, "alpha": self.aqs_cfg.alpha, "beta": self.aqs_cfg.beta,
                    "entropy_source": self.aqs_cfg.entropy_source, "formula": self.aqs_cfg.formula, "aqs": d}, tmp)
        os.replace(tmp, self.aqs_variant_path)
        vals = torch.tensor(list(d.values()), dtype=torch.float64)
        finite = vals[torch.isfinite(vals)]
        logger.info("AQS 변형 %s: n=%d mean %.4f std %.4f min %.4f max %.4f nan=%d → %s", self.aqs_cfg.variant, len(vals),
                    float(finite.mean()) if len(finite) else float("nan"), float(finite.std()) if len(finite) > 1 else 0.0,
                    float(finite.min()) if len(finite) else float("nan"), float(finite.max()) if len(finite) else float("nan"),
                    int((~torch.isfinite(vals)).sum()), self.aqs_variant_path)

    def setup(self, stage: Optional[str] = None) -> None:
        # 원본은 fbank 변환 전에 AQS 를 계산한다 — 피처 웜업(부모 setup) 전에 AQS 캐시를 채운다.
        if self.index_csv is None:
            self.outputs_directory.mkdir(parents=True, exist_ok=True)
            self.index_csv = self._default_index_path()
        if stage in (None, "fit"):
            self._precompute_aqs(self._build_index_dataset(splits=self.train_splits), "train")
            self._precompute_aqs(self._build_index_dataset(splits=self.val_splits), "val")
        if stage in (None, "test"):
            self._precompute_aqs(self._build_index_dataset(splits=self.test_splits), "test")
        super().setup(stage)

    def _loader(self, ds, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=self.num_workers,
                          collate_fn=qlung_collate, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers and self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds is None. Did you forget to call setup('fit')?")
        return self._loader(self.train_ds, self.shuffle_train, self.drop_last_train)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds is None. Did you forget to call setup('fit')?")
        return self._loader(self.val_ds, False, False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("test_ds is None. Did you forget to call setup('test')?")
        return self._loader(self.test_ds, False, False)
