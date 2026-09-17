# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Optional, Dict, Any, ClassVar, Mapping
from abc import ABC, abstractmethod
import pathlib
import logging

import torch

from rsc_toolkit.datasets.feature_cache import FeatureCache


logger = logging.getLogger(__name__)


class BaseAudioProcessor(ABC):
    """Feature front-end contract.

    Processors looked up through the registry (`rsc_toolkit.registries.PROCESSORS`, `data.params.features.kind`)
    declare two more things:

    - `is_spectrogram`: if True, the datamodule may apply SpecAugment to training batches
      (among built-ins only mel/fbank are True).
    - `from_options(opts, *, sampling_rate, feature_cache)`: builds the processor from the `features` config (dict).
      Each kind knows which keys it reads. Parameters specific to a new kind go into `features.params`.

    `get_processor_config()` is the **feature cache key** — as long as this dict stays the same, the cache
    still hits even if the construction path changes. Do not modify it.
    """

    is_spectrogram: ClassVar[bool] = False

    def __init__(
        self,
        feature_cache: Optional[FeatureCache] = None

    ) -> None:
        self.feature_cache = feature_cache

    @classmethod
    def from_options(cls, opts: Mapping[str, Any], *, sampling_rate: int, feature_cache: Optional[FeatureCache]):
        """Build the processor from the `features` config. Built-in kinds implement it; the default raises NotImplementedError."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_options — implement it to use this class as features.kind"
        )

    def get_processor_config(self) -> Dict[str, Any]:
        """Extract config that affects feature computation"""
        # Override in subclasses to return relevant config, like model name etc.
        # Exclude cache-related fields and private attributes
        return {}
    
    @abstractmethod
    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        """
        Convert audio to features.
        sample_id: Optional identifier for caching (e.g., "recording_123_cycle_0")
        """
        pass
    
    def make_features_cached(
        self, 
        audio: torch.Tensor, 
        sample_id: str
    ) -> torch.Tensor:
        """Make features with caching support"""
        if self.feature_cache and self.feature_cache.is_cached(sample_id):
            logger.debug(f"Cache hit for {sample_id}")
            cached_features = self.feature_cache.get(sample_id)
            if cached_features is not None:
                return cached_features
            logger.warning(f"Got cached features {cached_features}, recomputing.")
        
        # Compute features
        features = self.make_features(audio, sample_id=sample_id)
        
        # Cache if enabled
        if self.feature_cache:
            self.feature_cache.put(sample_id, features)
        
        return features
