"""
PyTorch Dataset and DataLoader integration for RSC datasets.

**Deprecated.** The training path is IndexDataset + IndexDataModule (rsc_toolkit/datasets/index_dataset.py,
rsc_toolkit/trainer/data/index_datamodule.py). This module is a parallel stack used only by its own unit tests, and
the ICBHI Crackle/Wheeze label rule is hard-coded in its 'Base' class, so it must not be used as the basis for new datasets. It will be removed in the next release.
"""
import warnings as _warnings

_warnings.warn(
    "rsc_toolkit.datasets.loaders is deprecated -- use IndexDataset/IndexDataModule (to be removed in the next release)",
    DeprecationWarning, stacklevel=2,
)
from typing import Literal, Optional, Callable, List, Tuple, Any
from abc import ABC, abstractmethod
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class BaseRSCPyTorchDataset(Dataset, ABC):
    """
    Base PyTorch Dataset for RSC tasks.
    Supports different loading strategies:
    - cycle_level: Each sample is a breathing cycle
    - clip_level: Each sample is a full recording clip
    """
    
    def __init__(
        self,
        samples: List[Any],
        sample_level: Literal["cycle", "clip"] = "cycle",
        feature_processor: Optional[Any] = None,  # BaseAudioProcessor
        use_cached_features: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        get_label_fn: Optional[Callable] = None,
    ):
        """
        Args:
            samples: List of samples (Cycle or Recording objects)
            sample_level: "cycle" for cycle-level samples, "clip" for full recordings
            feature_processor: Optional processor to extract features on-the-fly
            use_cached_features: If True, use cached features when available
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to labels
            get_label_fn: Function to extract label from sample (sample -> label)
        """
        self.samples = samples
        self.sample_level = sample_level
        self.feature_processor = feature_processor
        self.use_cached_features = use_cached_features
        self.transform = transform
        self.target_transform = target_transform
        self.get_label_fn = get_label_fn or self._default_get_label
        
        # Filter samples based on sample_level
        if sample_level == "cycle":
            self.samples = [s for s in samples if self._is_cycle(s)]
        elif sample_level == "clip":
            self.samples = [s for s in samples if self._is_recording(s)]
        else:
            raise ValueError(f"Unknown sample_level: {sample_level}")
    
    def _is_cycle(self, sample: Any) -> bool:
        """Check if sample is a Cycle"""
        return hasattr(sample, 'audio') and hasattr(sample, 'start') and hasattr(sample, 'end')
    
    def _is_recording(self, sample: Any) -> bool:
        """Check if sample is a Recording"""
        return hasattr(sample, 'wav_path') and hasattr(sample, 'meta')
    
    def _default_get_label(self, sample: Any) -> int:
        """Default label extraction - can be overridden"""
        if hasattr(sample, 'label') and sample.label is not None:
            return sample.label
        elif hasattr(sample, 'annotation'):
            # Extract label from annotation (e.g., ICBHI: Crackle + Wheeze)
            ann = sample.annotation
            # Support pandas Series, dict-like, or objects with attributes
            crackle = 0
            wheeze = 0
            if isinstance(ann, dict):
                crackle = int(ann.get('Crackle', 0))
                wheeze = int(ann.get('Wheeze', 0))
            else:
                if hasattr(ann, 'Crackle'):
                    crackle = int(getattr(ann, 'Crackle'))
                if hasattr(ann, 'Wheeze'):
                    wheeze = int(getattr(ann, 'Wheeze'))
                # Fallback for Series / mapping-like with key access
                try:
                    crackle = int(ann['Crackle']) if 'Crackle' in ann else crackle
                    wheeze = int(ann['Wheeze']) if 'Wheeze' in ann else wheeze
                except Exception:
                    pass
            # Combine: 0=Normal, 1=Crackle, 2=Wheeze, 3=Both
            return crackle + 2 * wheeze
        return 0  # Default to class 0
    
    def _load_features(self, sample: Any, sample_id: str) -> torch.Tensor:
        """Load features for a sample, using cache if available"""
        # Check if cached features exist
        # The BaseAudioProcessor cache attribute is named `feature_cache`.
        if self.use_cached_features and self.feature_processor:
            if self.feature_processor.feature_cache:
                if self.feature_processor.feature_cache.is_cached(sample_id):
                    return self.feature_processor.feature_cache.get(sample_id)
        
        # Extract audio
        if hasattr(sample, 'audio'):
            audio = sample.audio
        elif hasattr(sample, 'wav_path'):
            audio, _ = torchaudio.load(str(sample.wav_path))
        else:
            raise ValueError(f"Cannot extract audio from {type(sample)}")
        
        # Process with feature processor if available
        if self.feature_processor:
            if self.use_cached_features:
                features = self.feature_processor.make_features_cached(audio, sample_id)
            else:
                features = self.feature_processor.make_features(audio)
        else:
            # No processor - return raw audio
            features = audio

        # For per-sample embeddings, many processors return (1, D). Prefer (D,).
        if (
            self.feature_processor
            and isinstance(features, torch.Tensor)
            and features.dim() == 2
            and features.shape[0] == 1
        ):
            features = features.squeeze(0)
        
        return features
    
    def _get_sample_id(self, sample: Any) -> str:
        """Get unique ID for sample"""
        if hasattr(sample, 'wav_path'):
            return str(sample.wav_path.stem)
        elif hasattr(sample, 'start') and hasattr(sample, 'end'):
            # Cycle - use start/end times
            rec = getattr(sample, '_recording', None)
            if rec and hasattr(rec, 'wav_path'):
                return f"{rec.wav_path.stem}_cycle_{sample.start:.3f}_{sample.end:.3f}"
            return f"cycle_{sample.start:.3f}_{sample.end:.3f}"
        else:
            return str(hash(str(sample)))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get a sample (features, label)"""
        sample = self.samples[idx]
        sample_id = self._get_sample_id(sample)
        
        # Load features
        features = self._load_features(sample, sample_id)
        
        # Get label
        label = self.get_label_fn(sample)
        
        # Apply transforms
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        
        return features, label
    
    def __len__(self) -> int:
        return len(self.samples)


class ICBHIPyTorchDataset(BaseRSCPyTorchDataset):
    """
    PyTorch Dataset for ICBHI dataset.
    Works with Cycle or Recording objects.
    """
    pass  # Inherits all functionality from BaseRSCPyTorchDataset


def create_dataloader(
    dataset: BaseRSCPyTorchDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader from an RSC PyTorch Dataset.
    
    Args:
        dataset: RSC PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional arguments to DataLoader
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )
