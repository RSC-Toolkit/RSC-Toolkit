from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import importlib
import logging
import time

import lightning as L
import torch
from torch.utils.data import DataLoader

from rsc_toolkit.datasets.feature_cache import FeatureCache
from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset

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
        # raw/index locations
        project_root: str,
        raw_data_directory: str,
        outputs_directory: str,
        split_file: Optional[str] = None,
        # sampling/index
        sampling_rate: int = 16_000,
        sample_level: str = "cycle",
        index_filename: Optional[str] = None,
        # split names
        train_split: str = "train",
        val_split: str = "test",
        test_split: str = "test",
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

        self.project_root = Path(project_root)
        self.raw_data_directory = Path(raw_data_directory)
        self.outputs_directory = Path(outputs_directory)
        self.split_file = Path(split_file) if split_file else None

        self.sampling_rate = int(sampling_rate)
        self.sample_level = str(sample_level)
        self.index_filename = index_filename

        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.test_split = str(test_split)

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle_train = bool(shuffle_train)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)

        self.features_cfg = FeatureConfig(**(features or {}))
        self.feature_precompute = bool(feature_precompute)
        self.feature_precompute_batch_size = int(feature_precompute_batch_size)
        self.feature_precompute_num_workers = int(feature_precompute_num_workers)

        # set later
        self.index_csv: Optional[Path] = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

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
        return (self.outputs_directory / "dataset" / self.dataset_name / "index" / filename).resolve()

    def _build_index_dataset(self, *, split: str):
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
        )
        if self.split_file is not None:
            kwargs["split_file"] = self.split_file
        return self.index_dataset_cls(**kwargs)

    def _build_feature_processor(self, feature_cache: FeatureCache):
        if self.features_cfg.kind == "clap":    # TODO no construction here, either DI or factory
            from rsc_toolkit.processors.clap import CLAPProcessor
            return CLAPProcessor(
                sampling_rate=self.sampling_rate,
                model_name=self.features_cfg.clap_model_name,
                feature_cache=feature_cache,
            )
        if self.features_cfg.kind == "mel":
            from rsc_toolkit.processors.mel import MelBankAudioProcessor
            return MelBankAudioProcessor(
                sampling_rate=self.sampling_rate,
                n_mels=self.features_cfg.n_mels,
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

    def _warmup_feature_cache(self, ds: IndexFeatureDataset, *, split_name: str) -> None:
        fp = getattr(ds, "fp", None)
        cache = getattr(fp, "feature_cache", None)
        if cache is None:
            logger.warning("Feature precompute enabled but feature_cache is missing (split=%s)", split_name)
            return

        index_ds = getattr(ds, "index_ds", None)
        df = getattr(index_ds, "df", None)
        if df is None or "sample_id" not in df.columns:
            logger.warning("Cannot warm cache: missing index_ds.df['sample_id'] (split=%s)", split_name)
            return

        sample_ids = df["sample_id"].astype(str).tolist()
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
            def __init__(self, inner):
                self.inner = inner

            def __len__(self) -> int:
                return len(self.inner)

            def __getitem__(self, i: int) -> Optional[str]:
                sid = str(self.inner.df.iloc[int(i)]["sample_id"])
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
            _WarmupDataset(index_ds),
            batch_size=self.feature_precompute_batch_size,
            shuffle=False,
            num_workers=self.feature_precompute_num_workers,
            collate_fn=lambda batch: batch,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

        with torch.inference_mode():
            for batch in warmup_loader:
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
            self.train_ds = self._build_index_dataset(split=self.train_split)
            self.val_ds = self._build_index_dataset(split=self.val_split)

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
            self.test_ds = self._build_index_dataset(split=self.test_split)
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
