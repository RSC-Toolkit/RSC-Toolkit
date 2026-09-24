# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import ClassVar, Literal, Optional, List, Any, Type, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging
import functools
import pathlib
import hashlib
import json
import torch

import torchaudio

from rsc_toolkit.datasets.sample_index import REQUIRED_INDEX_COLUMNS as _REQUIRED_INDEX_COLUMNS


logger = logging.getLogger(__name__)


class BaseRSCDatasetDownloader(ABC, BaseModel):
    """
    Interface for downloading datasets
    """
    name: str
    outputs_directory: str
    force: bool = False

    @abstractmethod
    def validate_downloaded_files(self) -> bool:
        """
        Validate the downloaded files.
        """

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Download the dataset files
        """

    def download(self):
        outputs_path = pathlib.Path(self.outputs_directory).expanduser().resolve()
        logger.info("Dataset '%s' download directory: %s", self.name, outputs_path)

        if not self.force and self.validate_downloaded_files():
            logger.info(f"Found existing {self.name}, skipping download.")
            return
        logger.info("⬇️ Downloading %s", self.name)
        self.download_dataset()
        logger.info(f"✅ Completed {self.name} download")


class BaseRSCDatasetProcessor(ABC, BaseModel):
    """
    Interface for processing datasets for training
    """
    name: str # TODO better to use this from parent?
    outputs_directory: str
    force: bool = False
    preprocess_mode: Literal["full", "metadata_only", "none"] = "full"
    
    # Preprocessing options
    preprocess_features: bool = False
    feature_processor: Optional[Any] = None  # BaseAudioProcessor, but avoid circular import
    num_preprocess_workers: int = 4
    preprocess_batch_size: Optional[int] = None  # For GPU-based processors

    @functools.cached_property
    def cache_dir_path(self):

        def _create_unique_name(**kwargs):
            # Sort keys for determinism
            serialized = json.dumps(kwargs, sort_keys=True, default=str)
            # Hash for compact unique identifier
            return hashlib.md5(serialized.encode()).hexdigest()
        
        # Include feature processor config in cache key if present.
        #
        # The method is `get_processor_config()` (no leading underscore). Its presence is
        # checked with getattr; an AttributeError raised inside the method is not swallowed,
        # otherwise the config would silently drop out of the key (e.g. different n_mels sharing one cache).
        kwargs = {}
        if self.feature_processor is not None:
            getter = getattr(self.feature_processor, "get_processor_config", None)
            if callable(getter):
                kwargs["feature_processor"] = getter()
            else:
                logger.warning(
                    "%s has no get_processor_config(); its config is left out of the cache key. "
                    "Features with different configs may share the same cache.",
                    type(self.feature_processor).__name__,
                )

        key = _create_unique_name(
            preprocess_mode=self.preprocess_mode,
            **kwargs
        )

        cache_path = pathlib.Path(self.outputs_directory, "cache", key)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @functools.cached_property
    def index_dir_path(self):
        path = pathlib.Path(self.outputs_directory, "index")
        path.mkdir(parents=True, exist_ok=True)
        return path

    # sample_levels checked during index validation. Datasets that build only some of them override this.
    processed_sample_levels: Tuple[str, ...] = ("cycle", "clip")

    # Columns every index CSV must have (same as what IndexDataset requires).
    # The `split` column may be missing from older indexes, so it is not required.
    # Single source: rsc_toolkit.datasets.sample_index.REQUIRED_INDEX_COLUMNS (CSV_COLUMNS minus split)
    REQUIRED_INDEX_COLUMNS: ClassVar[Tuple[str, ...]] = _REQUIRED_INDEX_COLUMNS

    def processed_extra_files(self) -> List[str]:
        """
        Names of auxiliary artifacts that must sit next to the index in the index directory.
        Overridden per dataset (e.g. official_split.txt for sprsound).
        """
        return []

    def validate_index_labels(self, df: Any, *, sample_level: str) -> bool:
        """
        Label validation hook. The label space differs per dataset and task (ICBHI 4 classes, SPRSound
        7 event types, clip level sometimes unlabeled), so the default checks only the minimum:
        cycle-level rows must have no missing labels. Datasets override this for stricter rules.
        """
        if sample_level != "cycle":
            return True
        missing = int(df["label"].isna().sum())
        if missing:
            logger.error("cycle index has missing labels in %d/%d rows", missing, len(df))
            return False
        return True

    def _processed_index_dir(self) -> pathlib.Path:
        # index_dir_path creates the directory (cached_property), so validation only computes the path.
        return pathlib.Path(self.outputs_directory) / "index"

    def _validate_index_file(self, path: pathlib.Path, *, sample_level: str) -> bool:
        import pandas as pd

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.error("Cannot read index: %s (%s)", path, exc)
            return False

        missing_cols = [c for c in self.REQUIRED_INDEX_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.error("Index is missing columns %s: %s", missing_cols, path)
            return False

        if len(df) == 0:
            # Even if build-index exits 0, an index with 0 rows cannot be trained on.
            logger.error("Index has no rows: %s", path)
            return False

        levels = set(df["sample_level"].astype(str).str.strip().str.lower())
        if levels != {sample_level}:
            logger.error("Index %s has sample_level values %s (expected: %s)", path.name, sorted(levels), sample_level)
            return False

        stats_path = path.with_name(f"index_{sample_level}_stats.json")
        if stats_path.exists():
            try:
                total = json.loads(stats_path.read_text()).get("total_samples")
            except Exception as exc:
                logger.error("Cannot read stats file: %s (%s)", stats_path, exc)
                return False
            if total is not None and int(total) != len(df):
                logger.error("stats total_samples=%s differs from index row count %d (stale?): %s", total, len(df), stats_path)
                return False
        else:
            logger.warning("No stats file (optional): %s", stats_path)

        if not self.validate_index_labels(df, sample_level=sample_level):
            return False

        logger.info("Index OK: %s (%d rows)", path.name, len(df))
        return True

    def validate_processed_files(self) -> Optional[bool]:
        """
        Check that the preprocessing artifacts (indexes and auxiliary files) are intact.

        Returns:
          None  = no artifacts to check yet (build-index has not run). Not a failure.
          False = artifacts exist but are broken (missing columns, 0 rows, stats mismatch, missing auxiliary file, ...).
          True  = the indexes of all present levels are valid and the auxiliary files exist.

        Only levels that exist are checked, so a user who only uses cycle is not failed for
        not building the clip index. The feature cache has model/feature-specific shapes and is
        not checked here; use FeatureCache.validate_entries() for that.
        """
        index_dir = self._processed_index_dir()
        candidates = [(lvl, index_dir / f"index_{lvl}.csv") for lvl in self.processed_sample_levels]
        present = [(lvl, p) for lvl, p in candidates if p.exists()]
        if not present:
            logger.info("No processed artifacts (build-index has not run yet): %s", index_dir)
            return None

        ok = True
        for lvl, path in present:
            ok = self._validate_index_file(path, sample_level=lvl) and ok

        for name in self.processed_extra_files():
            extra = index_dir / name
            if not extra.exists() or extra.stat().st_size == 0:
                logger.error("Auxiliary artifact is missing or empty: %s", extra)
                ok = False

        return ok

    @abstractmethod
    def process_dataset(self) -> None:
        """
        Process the dataset for running
        """

    def preprocess_all_features(self, samples: List[Any]) -> None:
        """
        Preprocess all samples with slow feature extractor.
        This runs in parallel and caches results.
        """
        if not self.feature_processor or not self.preprocess_features:
            return
        
        from rsc_toolkit.datasets.feature_cache import FeatureCache
        from rsc_toolkit.datasets.preprocessing import ParallelPreprocessor

        # Attach a cache if there is none yet.
        #
        # BaseAudioProcessor exposes a single cache attribute, `feature_cache`
        # (there is no `cache_features` / `cache_dir` / `_init_cache()`).
        if self.feature_processor.feature_cache is None:
            cache = FeatureCache(cache_dir=self.cache_dir_path)
            cache.configure(
                processor_config=self.feature_processor.get_processor_config(),
                processor_name=type(self.feature_processor).__name__,
            )
            self.feature_processor.feature_cache = cache

        # Default to threads to avoid pickling issues with local functions
        # Threads work well for both CPU and GPU-bound tasks
        use_threads = True
        
        preprocessor = ParallelPreprocessor(
            num_workers=self.num_preprocess_workers,
            use_threads=use_threads,
            batch_size=self.preprocess_batch_size
        )
        
        def process_sample(sample, sample_id):
            # Extract audio from sample
            audio = self._extract_audio(sample)
            # Process with caching
            features = self.feature_processor.make_features_cached(
                audio, 
                sample_id=sample_id
            )
            return features
        
        def check_cached(sample_id):
            if self.feature_processor.feature_cache:
                return self.feature_processor.feature_cache.is_cached(sample_id)
            return False
        
        # Process all samples
        sample_ids = [self._get_sample_id(s) for s in samples]
        preprocessor.preprocess_samples(
            samples=samples,
            process_fn=process_sample,
            sample_ids=sample_ids,
            desc=f"Preprocessing {self.name} features",
            skip_cached=True,
            cache_check_fn=check_cached
        )
    
    def _extract_audio(self, sample: Any) -> torch.Tensor:
        """
        Extract audio tensor from sample.
        Override in subclasses if needed.
        """
        # Default implementation - subclasses should override
        if hasattr(sample, 'audio'):
            return sample.audio
        elif hasattr(sample, 'wav_path'):
            data, _ = torchaudio.load(str(sample.wav_path))
            return data
        else:
            raise NotImplementedError(f"Cannot extract audio from {type(sample)}")
    
    def _get_sample_id(self, sample: Any) -> str:
        """
        Get unique ID for sample (for caching).
        Override in subclasses.
        """
        if hasattr(sample, 'wav_path'):
            return str(sample.wav_path.stem)
        elif hasattr(sample, 'meta') and hasattr(sample.meta, 'patient_id'):
            return f"patient_{sample.meta.patient_id}"
        else:
            return str(hash(str(sample)))

    def process(self):
        if not self.force and self.validate_processed_files():
            logger.info(f"Found existing {self.name}, skipping processing.")
            return
        logger.info(f"⬇️ Processing {self.name}")
        self.process_dataset()
        logger.info(f"✅ Completed {self.name} processing")
    
class BaseRSCDataset(ABC, BaseModel):
    """
    Interface datasets
    """

    downloader_class: ClassVar[Type[BaseRSCDatasetDownloader]]
    processor_class: ClassVar[Type[BaseRSCDatasetProcessor]]

    downloader: BaseRSCDatasetDownloader
    processor: BaseRSCDatasetProcessor

    @classmethod
    def make_downloader(cls, **kwargs) -> BaseRSCDatasetDownloader:
        return cls.downloader_class(**kwargs)

    @classmethod
    def make_processor(cls, **kwargs) -> BaseRSCDatasetProcessor:
        return cls.processor_class(**kwargs)

    def download(self):
        self.downloader.download()

    def process(self):
        self.processor.process()
