from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

from torch.utils.data import DataLoader
import lightning as L
import torch
import time

from rsc_toolkit.datasets.toy_dataset import ICBHIDataset
from rsc_toolkit.datasets.index_dataset import IndexDataset, IndexFeatureDataset
from rsc_toolkit.datasets.feature_cache import FeatureCache
from rsc_toolkit.processors.clap import CLAPProcessor
from rsc_toolkit.processors.mel import MelBankAudioProcessor


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
    """
    Convert a list of dict samples (with keys like 'audio'/'features' + 'label')
    into a tuple batch: (x, y), compatible with your PLAdapter.

    It picks 'features' if present, else 'audio'.
    """
    # Choose input key
    x_key = "features" if "features" in batch[0] else "audio"

    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]

    # Stack tensors
    x = torch.stack(xs, dim=0)
    y = torch.as_tensor(ys, dtype=torch.long)

    return x, y


class ICBHIIndexDataModule(L.LightningDataModule):
    def __init__(
        self,
        # raw/index locations
        project_root: str,
        raw_data_directory: str,
        outputs_directory: str,
        split_file: str,

        # sampling/index
        sampling_rate: int = 16_000,
        sample_level: str = "cycle",  # "cycle" or "clip"
        index_filename: Optional[str] = None,  # override if you want

        # loaders
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,

        # processor behavior
        force_rebuild: bool = False,

        # feature caching
        features: Optional[dict] = None,
        feature_precompute: bool = False,
        feature_precompute_batch_size: int = 8,
        feature_precompute_num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()

        self.project_root = Path(project_root)
        self.raw_data_directory = Path(raw_data_directory)
        self.outputs_directory = Path(outputs_directory)
        self.split_file = Path(split_file)

        self.sampling_rate = sampling_rate
        self.sample_level = sample_level
        self.index_filename = index_filename  # e.g. "index_cycle.csv"
        self.force_rebuild = force_rebuild

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.features_cfg = FeatureConfig(**(features or {}))
        self.feature_precompute = feature_precompute
        self.feature_precompute_batch_size = feature_precompute_batch_size
        self.feature_precompute_num_workers = feature_precompute_num_workers

        # set later
        self.processor = None
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
            def __init__(self, inner: ICBHIIndexDataset):
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
                "Feature cache warmup split=%s using num_workers=%d (may require the feature processor to be picklable)",
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
        """
        Build datasets in-memory here. Called on every process in DDP.
        """
        # Ensure prepare_data ran and set index_csv
        if self.index_csv is None:
            # In case someone calls setup without prepare_data (rare), rebuild minimal paths
            out_dir = self.outputs_directory
            out_dir.mkdir(parents=True, exist_ok=True)
            self.index_csv = out_dir / "dataset" / "ICBHI" / "index" / f"index_{self.sample_level}.csv"

        base_kwargs = dict(
            index_csv=self.index_csv,
            data_root=self.raw_data_directory,
            sampling_rate=self.sampling_rate,
            sample_level=self.sample_level,
            split_file=self.split_file,
        )

        # Build base audio datasets
        if stage in (None, "fit"):
            self.train_ds = ICBHIIndexDataset(**base_kwargs, split="train")
            self.val_ds = ICBHIIndexDataset(**base_kwargs, split="test")

            if self.features_cfg.enabled:
                self.train_ds = self._wrap_with_features(self.train_ds)
                self.val_ds = self._wrap_with_features(self.val_ds)

            if self.features_cfg.enabled and self.feature_precompute:
                if not self._is_global_zero():
                    logger.info("Skipping feature cache warmup on non-global-zero rank")
                else:
                    self._warmup_feature_cache(self.train_ds, split_name="train")  # type: ignore[arg-type]
                    self._warmup_feature_cache(self.val_ds, split_name="val")  # type: ignore[arg-type]

        # TODO need to sort out the val==test case
        # if stage in (None, "test"):
        #     self.test_ds = ICBHIIndexDataset(**base_kwargs, split="test")
        #     if self.features_cfg.enabled:
        #         self.test_ds = self._wrap_with_features(self.test_ds)

    def _build_feature_processor(self, feature_cache: FeatureCache):
        if self.features_cfg.kind == "clap":
            return CLAPProcessor(
                sampling_rate=self.sampling_rate,
                model_name=self.features_cfg.clap_model_name,
                feature_cache=feature_cache,
            )
        if self.features_cfg.kind == "mel":
            return MelBankAudioProcessor(
                sampling_rate=self.sampling_rate,
                n_mels=self.features_cfg.n_mels,
                feature_cache=feature_cache,
            )
        raise ValueError(f"Unknown features.kind: {self.features_cfg.kind}")

    def _wrap_with_features(self, base_ds):
        out_dir = self.outputs_directory
        cache_dir = Path(self.features_cfg.cache_dir) if self.features_cfg.cache_dir else (out_dir / "features_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        feature_cache = FeatureCache(cache_dir=cache_dir)
        fp = self._build_feature_processor(feature_cache)
        feature_cache.configure(fp.get_processor_config(), processor_name=fp.__class__.__name__)

        return IndexFeatureDataset(base_ds, fp)

    # ---- dataloaders ----

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
