from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple, Sequence, Union
import importlib
import logging
import time

from torch.utils.data import DataLoader, ConcatDataset
import torch
import lightning as L
import tqdm

from rsc_toolkit.datasets.feature_cache import FeatureCache
from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset
from rsc_toolkit.datasets.paths import default_dataset_dir
from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    enabled: bool = False
    cache_dir: Optional[str] = None
    kind: str = "clap"  # registries.PROCESSORS 의 이름: clap | mel | fbank | sclmeta_mel | waveform | (플러그인)
    # clap params
    clap_model_name: str = "laion/clap-htsat-unfused"
    # mel params
    n_mels: int = 64
    target_frames: int = 1024
    frame_shift: int = 10
    window_type: str = "hanning"
    dither: float = 0.0
    normalize: bool = True
    per_sample_stats: bool = False  # 비-AST 계열 논문 재현용 (샘플별 mean/std)
    # sclmeta_mel (torchaudio MelSpectrogram) 파라미터
    n_fft: int = 1024
    win_length: Optional[int] = None
    hop_length: int = 512
    f_min: float = 50.0
    f_max: float = 2000.0
    # 새 kind(플러그인) 전용 파라미터 — 위 flat 필드는 내장 kind 하위호환용이고 더 늘리지 않는다.
    # from_options 에는 flat 필드와 params 를 합친 dict 가 들어간다 (params 가 우선).
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpecAugmentOptions:
    enabled: bool = False
    policy: str = "icbhi_ast_sup"
    mask: str = "mean"


from rsc_toolkit.datasets import collate as _collate  # P2-6: collate 단일 구현

# IndexDataModule 기본 collate: (x, y[, meta long]) — 4번째 원소가 있을 때만 meta (sgscl 계열)
_tuple_collate = _collate.tuple_collate


def _import_from_path(dotted_path: str) -> Any:
    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid dotted path: {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class IndexDataModule(L.LightningDataModule):
    """
    Generic index-based LightningDataModule.

    Provide an `index_dataset_cls` (dotted import path) that builds samples from an index CSV
    and returns dicts containing at least: {"audio": Tensor, "label": Optional[int], "sample_id": str}.

    batch[2](meta) 사이드채널 선언: `meta_kind` 는 이 datamodule 이 배치 3번째 원소로 무엇을 보내는지
    이름 붙인 것이다 (범용 경로는 `meta_label_mode` 그대로; 서브클래스는 클래스 속성으로 덮어쓴다 —
    예: "qlung_aqs", "pcmcl_ssl_label", "bts_meta7"). 모델 쪽 `accepts_meta` 와 어댑터 setup 에서 대조된다.

    확장 통로 (전부 additive — 기본값이면 예전 코드 경로와 동일):
    - `index_dataset_params`: 커스텀 `index_dataset_cls` 생성자에 그대로 넘길 추가 인자 dict
      (예전에는 클래스 속성 주입·메서드 복붙·`__new__` 바꿔치기로 우회했다).
    - `collate_fn`: dotted path 또는 callable — 세 dataloader 의 collate 를 바꾼다 (기본 `_tuple_collate`).
    - `dataset_wrapper_cls`: dotted path 또는 클래스 — 피처 래퍼(`IndexFeatureDataset`) 교체.
      `(index_ds, feature_processor, feature_transform=, meta_label_fn=)` 시그니처를 따라야 한다.
    서브클래스용 훅: `_index_dataset_kwargs(split)` 로 생성 인자를 더하고, `_wrap_with_features(..., split_role=)`
    로 train/val/test 역할을 명시적으로 받는다 (`base_ds is self.train_ds` identity 비교 불필요).
    """

    @property
    def meta_kind(self) -> Optional[str]:
        return self.meta_label_mode or None

    def __init__(
        self,
        # dataset identity
        dataset_name: str,
        index_dataset_cls: str, # TODO class or import name
        # data location
        project_root: str,
        data_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        # sampling/index
        sampling_rate: int = 16_000,
        sample_level: str = "cycle",
        index_filename: Optional[str] = None,
        desired_length_sec: Optional[float] = None,
        pad_type: str = "repeat",
        # split names (supports arbitrary split labels, including numbers)
        train_splits: Sequence[Any] = ("train",),
        val_splits: Sequence[Any] = ("test",),
        test_splits: Sequence[Any] = ("test",),
        # loaders
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        # feature caching
        features: Optional[dict] = None,
        feature_precompute: bool = False,
        feature_precompute_batch_size: int = 8,
        feature_precompute_num_workers: int = 0,
        specaugment: Optional[dict] = None,
        meta_label_mode: Optional[str] = None,
        fade: bool = True,
        resample_after_slice: bool = False,
        time_unit: str = "s",
        raw_data_subdir: Optional[str] = None,   # 원음이 data_dir 하위 폴더에 있을 때 (예: SPRSound 'Classification')
        # 확장 통로 (P1-9, additive)
        index_dataset_params: Optional[dict] = None,
        collate_fn: Optional[Union[str, Callable]] = None,
        dataset_wrapper_cls: Optional[Union[str, type]] = None,
    ):
        super().__init__()

        self.dataset_name = str(dataset_name)

        if isinstance(index_dataset_cls, (str, Path)):
            index_dataset_cls_path = str(index_dataset_cls)
            self.index_dataset_cls = _import_from_path(index_dataset_cls_path)
        else:
            self.index_dataset_cls = index_dataset_cls

        self.project_root = Path(project_root).expanduser().resolve()
        if data_dir is None:
            self.data_dir = default_dataset_dir(self.dataset_name)
        else:
            self.data_dir = self._resolve_path(data_dir)

        self.raw_data_subdir = raw_data_subdir
        self.raw_data_directory = self._resolve_raw_data_directory()
        if raw_data_subdir:
            self.raw_data_directory = (self.raw_data_directory / raw_data_subdir).resolve()
        self.outputs_directory = self.data_dir
        self.split_file = self._resolve_path(split_file) if split_file else None

        self.sampling_rate = int(sampling_rate)
        self.sample_level = str(sample_level)
        self.index_filename = index_filename
        self.desired_length_sec = float(desired_length_sec) if desired_length_sec is not None else None
        self.pad_type = str(pad_type)
        self.audio_preprocess_cfg = AudioPreprocessConfig(
            desired_length_sec=self.desired_length_sec,
            pad_type=self.pad_type,
        )
        self.audio_preprocessor = AudioPreprocessor(self.sampling_rate, self.audio_preprocess_cfg)

        self.train_splits = self._normalize_splits(train_splits, role="train")
        self.val_splits = self._normalize_splits(val_splits, role="val")
        self.test_splits = self._normalize_splits(test_splits, role="test")

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle_train = bool(shuffle_train)
        self.drop_last_train = bool(drop_last_train)  # 논문 프로토콜 재현용 (원본 train loader drop_last=True)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)

        self.features_cfg = FeatureConfig(**(features or {}))
        if self.features_cfg.cache_dir:
            self.features_cfg.cache_dir = str(self._resolve_feature_cache_dir(self.features_cfg.cache_dir))
        self.feature_precompute = bool(feature_precompute)
        self.feature_precompute_batch_size = int(feature_precompute_batch_size)
        self.feature_precompute_num_workers = int(feature_precompute_num_workers)
        self.specaugment_cfg = SpecAugmentOptions(**(specaugment or {}))
        # sgscl 계열(DAT·SG-SCL)용 meta 라벨 (예: "dev" = ICBHI 청진기 4종).
        # 원본처럼 train 배치에만 (x, y, meta) 로 붙는다.
        self.meta_label_mode = meta_label_mode
        self.fade = bool(fade)
        self.resample_after_slice = bool(resample_after_slice)
        self.time_unit = str(time_unit)

        self.index_dataset_params: Dict[str, Any] = dict(index_dataset_params or {})
        self.collate_fn: Callable = (
            _import_from_path(collate_fn) if isinstance(collate_fn, str) else (collate_fn or _tuple_collate)
        )
        self.dataset_wrapper_cls = (
            _import_from_path(dataset_wrapper_cls) if isinstance(dataset_wrapper_cls, str) else dataset_wrapper_cls
        )

        # set later
        self.index_csv: Optional[Path] = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _resolve_raw_data_directory(self) -> Path:
        """
        Resolve where the raw audio files live.

        By default this matches `data_dir`, but for bundled fixture datasets (e.g. `toy`)
        we fall back to the repository `sample_data` if `data_dir` doesn't contain audio.
        """
        data_dir = self.data_dir

        if self.dataset_name.lower() == "toy":
            try:
                has_wavs = data_dir.exists() and any(data_dir.glob("*.wav"))
            except Exception:
                has_wavs = False

            if not has_wavs:
                bundled = (self.project_root / "sample_data" / "toy_dataset" / "files").resolve()
                if bundled.exists():
                    return bundled

        return data_dir

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

    @staticmethod
    def _normalize_splits(
        splits: Optional[Sequence[Any]],
        *,
        role: str,
    ) -> List[str]:
        if splits is None:
            raise ValueError(f"{role}_splits is None; provide at least one split label")

        # yaml might pass a scalar when a list was intended; treat it as a singleton list
        if isinstance(splits, str):
            splits_list: List[Any] = [splits]
        elif isinstance(splits, (list, tuple, set)):
            splits_list = list(splits)
        else:
            splits_list = [splits]

        out: List[str] = []
        for split in splits_list:
            s = str(split).strip()
            if not s:
                continue
            out.append(s.lower())

        if not out:
            raise ValueError(f"{role}_splits is empty; provide at least one split label")

        # preserve order but drop duplicates
        deduped: List[str] = []
        seen = set()
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)
        return deduped

    def _is_global_zero(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            try:
                return bool(trainer.is_global_zero)
            except Exception:
                pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                return True
        return True

    def _default_index_path(self) -> Path:
        filename = self.index_filename or f"index_{self.sample_level}.csv"
        return (self.outputs_directory / "index" / filename).resolve()

    def _index_dataset_kwargs(self, *, split: str, role: Optional[str] = None) -> Dict[str, Any]:
        """
        `index_dataset_cls(**kwargs)` 의 인자 조립 — 서브클래스는 이것을 override 해 dict 에 더하기만 하면 된다
        (예전엔 이 블록을 통째로 복붙했다). config 의 `index_dataset_params` 가 마지막에 덮어쓴다.
        `role` ∈ {"train","val","test"} 은 setup 이 넘긴다 — 학습용 데이터셋만 다르게 만들 때(ProtoNet 크롭, DT 증강·splice) 쓴다.
        """
        kwargs: Dict[str, Any] = dict(
            index_csv=self.index_csv,
            data_root=self.raw_data_directory,
            sampling_rate=self.sampling_rate,
            sample_level=self.sample_level,
            split=split,
            audio_preprocessor=self.audio_preprocessor,
        )
        # 기본값이 아닐 때만 넘긴다 — 커스텀 index_dataset_cls 가 이 인자를 모를 수 있다
        if not self.fade:
            kwargs["fade"] = False
        if self.resample_after_slice:
            kwargs["resample_after_slice"] = True
        if self.time_unit != "s":
            kwargs["time_unit"] = self.time_unit
        if self.split_file is not None:
            kwargs["split_file"] = self.split_file
        kwargs.update(self.index_dataset_params)
        return kwargs

    def _build_index_dataset_single(self, *, split: str, role: Optional[str] = None):
        if self.index_csv is None:
            self.index_csv = self._default_index_path()
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index csv not found: {self.index_csv}")
        return self.index_dataset_cls(**self._index_dataset_kwargs(split=split, role=role))

    def _build_index_dataset(self, *, splits: Sequence[str], role: Optional[str] = None):
        splits = list(splits)
        if len(splits) == 1:
            return self._build_index_dataset_single(split=splits[0], role=role)
        datasets = [self._build_index_dataset_single(split=s, role=role) for s in splits]
        return ConcatDataset(datasets)

    def _feature_options(self) -> Dict[str, Any]:
        """from_options 에 넘길 dict: FeatureConfig 의 flat 필드 + params (params 가 우선)."""
        opts = asdict(self.features_cfg)
        params = opts.pop("params", None) or {}
        opts.update(params)
        return opts

    def _feature_processor_cls(self):
        """features.kind → 프로세서 클래스 (registries.PROCESSORS). 미지 kind 는 가용 목록과 함께 KeyError."""
        from rsc_toolkit.registries import PROCESSORS

        try:
            return PROCESSORS.get(self.features_cfg.kind)
        except KeyError:
            raise KeyError(
                f"Unknown features.kind '{self.features_cfg.kind}' (data.params.features.kind). "
                f"Available: {sorted(PROCESSORS)}"
            ) from None

    def _build_feature_processor(self, feature_cache: FeatureCache):
        # 예전의 kind 별 if-elif 는 각 프로세서의 from_options 로 옮겼다 (생성 인자 동일 → 캐시 키 동일).
        cls = self._feature_processor_cls()
        return cls.from_options(self._feature_options(), sampling_rate=self.sampling_rate, feature_cache=feature_cache)

    def _make_feature_processor_with_cache(self):
        """피처 캐시(디렉터리·키·체인 대장)를 구성한 프로세서 — 래퍼 종류와 무관한 공통 부분."""
        cache_dir = (
            Path(self.features_cfg.cache_dir)
            if self.features_cfg.cache_dir
            else (self.outputs_directory / "features_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        feature_cache = FeatureCache(cache_dir=cache_dir)
        fp = self._build_feature_processor(feature_cache)
        feature_cache.configure(fp.get_processor_config(), processor_name=fp.__class__.__name__,
                                audio_chain=self._audio_chain())
        return fp

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None, split_role: Optional[str] = None):
        """
        인덱스 데이터셋을 피처 래퍼로 감싼다. `split_role` ∈ {"train","val","test"} 는 setup 이 넘긴다 —
        서브클래스는 `base_ds is self.train_ds` 같은 identity 비교 대신 이것을 보라.
        """
        fp = self._make_feature_processor_with_cache()
        wrapper = self.dataset_wrapper_cls or IndexFeatureDataset
        return wrapper(base_ds, fp, feature_transform=feature_transform, meta_label_fn=meta_label_fn)

    def _resolve_feature_cache_dir(self, cache_dir: str) -> "Path":
        """
        피처 캐시 위치. config 의 상대 경로는 프로젝트 루트 기준이지만,
        환경변수 **`RSCTK_FEATURE_CACHE_ROOT`** 가 있으면 그 아래로 옮겨 붙인다.

        캐시는 수십 GB 라 저장소 트리 밖(다른 볼륨)에 두고 싶을 때가 있는데,
        config 를 열 몇 개씩 고치면 재현 기록이 지저분해진다. 경로만 갈아끼우는 이 방법이면
        **config 는 그대로 두고** 환경변수 하나로 옮길 수 있다(디렉터리 이름은 그대로라 이동도 `mv` 한 번).
        절대 경로가 config 에 박혀 있으면 그대로 존중한다.
        """
        import os

        root = os.environ.get("RSCTK_FEATURE_CACHE_ROOT")
        p = Path(cache_dir)
        if root and not p.is_absolute():
            return (Path(root).expanduser() / p).resolve()
        return self._resolve_path(cache_dir)

    def _audio_chain(self) -> dict:
        """
        피처 캐시 키에 들어가지 않는 **앞단 오디오 체인**. FeatureCache 가 대장에 남겨
        같은 데이터셋을 서로 다른 전처리로 캐시하는 사고를 막는다(feature_cache.configure 주석 참조).
        """
        return {
            "dataset_name": getattr(self, "dataset_name", "?"),
            "sampling_rate": getattr(self, "sampling_rate", None),
            "sample_level": getattr(self, "sample_level", None),
            "desired_length_sec": getattr(self, "desired_length_sec", None),
            "pad_type": getattr(self, "pad_type", None),
            "fade": getattr(self, "fade", None),
            "resample_after_slice": getattr(self, "resample_after_slice", None),
            "time_unit": getattr(self, "time_unit", None),
        }

    def _build_feature_transform(self, *, training: bool):
        if not training:
            return None
        if not self.specaugment_cfg.enabled:
            return None
        # 스펙트로그램 kind 에만 건다 — 예전 {"mel","fbank"} 화이트리스트와 같은 진리표이지만
        # 이제 플러그인 kind 도 is_spectrogram=True 선언만으로 통과한다.
        if not getattr(self._feature_processor_cls(), "is_spectrogram", False):
            return None
        from rsc_toolkit.registries import AUGMENTATIONS

        return AUGMENTATIONS.get("specaugment")(policy=self.specaugment_cfg.policy, mask=self.specaugment_cfg.mask)

    @staticmethod
    def _collect_sample_ids(index_ds: Any) -> List[str]:
        if isinstance(index_ds, ConcatDataset):
            sample_ids: List[str] = []
            for sub in index_ds.datasets:
                sample_ids.extend(IndexDataModule._collect_sample_ids(sub))
            return sample_ids

        df = getattr(index_ds, "df", None)
        if df is not None and getattr(df, "columns", None) is not None and "sample_id" in df.columns:
            return df["sample_id"].astype(str).tolist()

        # Fallback: compute by indexing the dataset (may be slow but keeps this generic).
        out: List[str] = []
        for i in range(len(index_ds)):
            item = index_ds[int(i)]
            out.append(str(item["sample_id"]))
        return out

    def _warmup_feature_cache(self, ds: IndexFeatureDataset, *, split_name: str) -> None:
        fp = getattr(ds, "fp", None)
        cache = getattr(fp, "feature_cache", None)
        if cache is None:
            logger.warning("Feature precompute enabled but feature_cache is missing (split=%s)", split_name)
            return

        index_ds = getattr(ds, "index_ds", None)
        if index_ds is None:
            logger.warning("Cannot warm cache: missing ds.index_ds (split=%s)", split_name)
            return

        sample_ids = self._collect_sample_ids(index_ds)
        total = len(sample_ids)
        cached_before = sum(1 for sid in sample_ids if cache.is_cached(sid))
        missing = total - cached_before

        try:
            stats = cache.get_cache_stats()
            cache_dir = stats.get("cache_dir")
        except Exception:
            cache_dir = None

        logger.info(
            "Feature cache warmup split=%s: cached=%d/%d missing=%d cache_dir=%s",
            split_name,
            cached_before,
            total,
            missing,
            cache_dir,
        )

        if missing == 0:
            return

        start = time.time()
        computed = 0
        log_every = max(25, missing // 10)

        class _WarmupDataset(torch.utils.data.Dataset):
            def __init__(self, inner, sample_ids: Sequence[str]):
                self.inner = inner
                self.sample_ids = list(sample_ids)

            def __len__(self) -> int:
                return len(self.inner)

            def __getitem__(self, i: int) -> Optional[str]:
                sid = str(self.sample_ids[int(i)])
                if cache.is_cached(sid):
                    return None
                item = self.inner[int(i)]
                audio = item["audio"].squeeze()
                fp.make_features_cached(audio, sample_id=sid)
                return sid

        if self.feature_precompute_num_workers > 0:
            logger.info(
                "Feature cache warmup split=%s using num_workers=%d (feature processor may need to be picklable)",
                split_name,
                self.feature_precompute_num_workers,
            )

        warmup_loader = DataLoader(
            _WarmupDataset(index_ds, sample_ids),
            batch_size=self.feature_precompute_batch_size,
            shuffle=False,
            num_workers=self.feature_precompute_num_workers,
            collate_fn=lambda batch: batch,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

        with torch.inference_mode():
            for batch in tqdm.tqdm(warmup_loader):
                computed += sum(1 for sid in batch if sid is not None)
                if computed > 0 and computed % log_every == 0:
                    elapsed = time.time() - start
                    logger.info(
                        "Feature cache warmup split=%s: computed=%d/%d (%.1fs)",
                        split_name,
                        computed,
                        missing,
                        elapsed,
                    )

        elapsed = time.time() - start
        cached_after = sum(1 for sid in sample_ids if cache.is_cached(sid))
        logger.info(
            "Feature cache warmup done split=%s: computed=%d elapsed=%.1fs now_cached=%d/%d",
            split_name,
            computed,
            elapsed,
            cached_after,
            total,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.index_csv is None:
            self.outputs_directory.mkdir(parents=True, exist_ok=True)
            self.index_csv = self._default_index_path()

        if stage in (None, "fit"):
            self.train_ds = self._build_index_dataset(splits=self.train_splits, role="train")
            self.val_ds = self._build_index_dataset(splits=self.val_splits, role="val")

            if self.features_cfg.enabled:
                train_transform = self._build_feature_transform(training=True)
                meta_label_fn = None
                if self.meta_label_mode is not None:
                    from rsc_toolkit.datasets.meta_labels import build_meta_label_fn

                    meta_label_fn = build_meta_label_fn(self.meta_label_mode)
                self.train_ds = self._wrap_with_features(
                    self.train_ds, feature_transform=train_transform, meta_label_fn=meta_label_fn, split_role="train"
                )
                self.val_ds = self._wrap_with_features(self.val_ds, feature_transform=None, split_role="val")

            if self.features_cfg.enabled and self.feature_precompute:
                if not self._is_global_zero():
                    logger.info("Skipping feature cache warmup on non-global-zero rank")
                else:
                    self._warmup_feature_cache(self.train_ds, split_name="train")  # type: ignore[arg-type]
                    self._warmup_feature_cache(self.val_ds, split_name="val")  # type: ignore[arg-type]

        if stage in (None, "test"):
            self.test_ds = self._build_index_dataset(splits=self.test_splits, role="test")
            if self.features_cfg.enabled:
                self.test_ds = self._wrap_with_features(self.test_ds, split_role="test")
            if self.features_cfg.enabled and self.feature_precompute:
                if not self._is_global_zero():
                    logger.info("Skipping feature cache warmup on non-global-zero rank")
                else:
                    self._warmup_feature_cache(self.test_ds, split_name="test")  # type: ignore[arg-type]

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds is None. Did you forget to call setup('fit')?")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds is None. Did you forget to call setup('fit')?")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("test_ds is None. Did you forget to call setup('test')?")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
