from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import pathlib
import logging

import torch

from rsc_toolkit.datasets.feature_cache import FeatureCache


logger = logging.getLogger(__name__)


class BaseAudioProcessor(ABC):

    def __init__(
        self,
        feature_cache: Optional[FeatureCache] = None

    ) -> None:    
        self.feature_cache = feature_cache

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
