from typing import ClassVar, Literal, Optional, List, Any, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging
import functools
import pathlib
import hashlib
import json
import torch


logger = logging.getLogger(__name__)


class BaseRSCDatasetDownloader(ABC, BaseModel):
    """
    Interface for downloading datasets
    """
    name: str
    outputs_directory: str
    force: bool = False

    @abstractmethod
    def validate_dataset_files(self) -> bool:
        """
        Validate the dataset files.
        """

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Download the dataset files
        """

    def download(self):
        if not self.force and self.validate_dataset_files():
            logger.info(f"Found existing {self.name}, skipping download.")
            return
        logger.info(f"⬇️ Downloading {self.name}")
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
        
        # Include feature processor config in cache key if present
        kwargs = {}
        if self.feature_processor:
            try:
                proc_config = self.feature_processor._get_processor_config()
                kwargs["feature_processor"] = proc_config
            except AttributeError:
                pass
        
        key = _create_unique_name(
            preprocess_mode=self.preprocess_mode,
            **kwargs
        )

        # create cache/dataset/DatasetName dir for temp files.
        cache_path = pathlib.Path(self.outputs_directory, "dataset", self.name, key)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @abstractmethod
    def validate_dataset_files(self) -> bool:
        """
        Validate preocessed files files
        """

    @abstractmethod
    def process_dataset(self) -> None:
        """
        Process the dataset for running
        """
    
    def preprocess_all_features(self, samples: List[Any]) -> None:
        """
        Preprocess all samples with expensive feature extractor.
        This runs in parallel and caches results.
        """
        if not self.feature_processor or not self.preprocess_features:
            return
        
        from rsc_toolkit.datasets.preprocessing import ParallelPreprocessor
        
        # Enable caching in processor if not already enabled
        if not self.feature_processor.cache_features:
            self.feature_processor.cache_features = True
            self.feature_processor.cache_dir = str(self.cache_dir_path)
            # Re-initialize cache
            self.feature_processor._init_cache()
        
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
            if self.feature_processor._feature_cache:
                return self.feature_processor._feature_cache.is_cached(sample_id)
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
            # Load from file
            import torchaudio
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
        if not self.force and self.validate_dataset_files():
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


