from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Sequence, Union
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
    kind: str = "clap"  # "clap" or "mel"
    # clap params
    clap_model_name: str = "laion/clap-htsat-unfused"
    # mel params
    n_mels: int = 64
    target_frames: int = 1024
    frame_shift: int = 10
    window_type: str = "hanning"
    dither: float = 0.0
    normalize: bool = True


def _tuple_collate(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    x = torch.stack(xs, dim=0)
    y = torch.as_tensor(ys, dtype=torch.long)
    return x, y


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
    """

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
        pin_memory: bool = False,
        persistent_workers: bool = False,
        # feature caching
        features: Optional[dict] = None,
        feature_precompute: bool = False,
        feature_precompute_batch_size: int = 8,
        feature_precompute_num_workers: int = 0,
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

        self.raw_data_directory = self._resolve_raw_data_directory()
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
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)

        self.features_cfg = FeatureConfig(**(features or {}))
        if self.features_cfg.cache_dir:
            self.features_cfg.cache_dir = str(self._resolve_path(self.features_cfg.cache_dir))
        self.feature_precompute = bool(feature_precompute)
        self.feature_precompute_batch_size = int(feature_precompute_batch_size)
        self.feature_precompute_num_workers = int(feature_precompute_num_workers)

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

    def _build_index_dataset_single(self, *, split: str):
        if self.index_csv is None:
            self.index_csv = self._default_index_path()
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index csv not found: {self.index_csv}")

        kwargs: Dict[str, Any] = dict(
            index_csv=self.index_csv,
            data_root=self.raw_data_directory,
            sampling_rate=self.sampling_rate,
            sample_level=self.sample_level,
            split=split,
            audio_preprocessor=self.audio_preprocessor,
        )
        if self.split_file is not None:
            kwargs["split_file"] = self.split_file
        return self.index_dataset_cls(**kwargs)

    def _build_index_dataset(self, *, splits: Sequence[str]):
        splits = list(splits)
        if len(splits) == 1:
            return self._build_index_dataset_single(split=splits[0])
        datasets = [self._build_index_dataset_single(split=s) for s in splits]
        return ConcatDataset(datasets)

    def _build_feature_processor(self, feature_cache: FeatureCache):
        if self.features_cfg.kind == "clap":    # TODO no construction here, either DI or factory
            from rsc_toolkit.processors.clap import CLAPProcessor
            return CLAPProcessor(
                sampling_rate=self.sampling_rate,
                model_name=self.features_cfg.clap_model_name,
                feature_cache=feature_cache,
            )
        if self.features_cfg.kind in {"mel", "fbank"}:
            from rsc_toolkit.processors.mel import MelBankAudioProcessor
            return MelBankAudioProcessor(
                sampling_rate=self.sampling_rate,
                n_mels=self.features_cfg.n_mels,
                target_frames=self.features_cfg.target_frames,
                frame_shift=self.features_cfg.frame_shift,
                window_type=self.features_cfg.window_type,
                dither=self.features_cfg.dither,
                normalize=self.features_cfg.normalize,
                feature_cache=feature_cache,
            )
        raise ValueError(f"Unknown features.kind: {self.features_cfg.kind}")

    def _wrap_with_features(self, base_ds):
        cache_dir = (
            Path(self.features_cfg.cache_dir)
            if self.features_cfg.cache_dir
            else (self.outputs_directory / "features_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        feature_cache = FeatureCache(cache_dir=cache_dir)
        fp = self._build_feature_processor(feature_cache)
        feature_cache.configure(fp.get_processor_config(), processor_name=fp.__class__.__name__)

        return IndexFeatureDataset(base_ds, fp)

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
            self.train_ds = self._build_index_dataset(splits=self.train_splits)
            self.val_ds = self._build_index_dataset(splits=self.val_splits)

            if self.features_cfg.enabled:
                self.train_ds = self._wrap_with_features(self.train_ds)
                self.val_ds = self._wrap_with_features(self.val_ds)

            if self.features_cfg.enabled and self.feature_precompute:
                if not self._is_global_zero():
                    logger.info("Skipping feature cache warmup on non-global-zero rank")
                else:
                    self._warmup_feature_cache(self.train_ds, split_name="train")  # type: ignore[arg-type]
                    self._warmup_feature_cache(self.val_ds, split_name="val")  # type: ignore[arg-type]

        if stage in (None, "test"):
            self.test_ds = self._build_index_dataset(splits=self.test_splits)
            if self.features_cfg.enabled:
                self.test_ds = self._wrap_with_features(self.test_ds)
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
            num_workers=self.num_workers,
            collate_fn=_tuple_collate,
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
            collate_fn=_tuple_collate,
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
            collate_fn=_tuple_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
