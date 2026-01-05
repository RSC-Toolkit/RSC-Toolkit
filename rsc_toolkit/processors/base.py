from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator, ConfigDict
import torch
import pathlib
import logging


from rsc_toolkit.augmentations import Augmentation


logger = logging.getLogger(__name__)


class BaseAudioProcessor(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    augmentations: Optional[Augmentation] = None
    cache_features: bool = False
    cache_dir: Optional[str] = None
    _feature_cache: Optional[Any] = None  # FeatureCache, but avoid circular import
    
    @model_validator(mode='after')
    def _init_cache(self):
        """Initialize feature cache if caching is enabled"""
        if self.cache_features and self.cache_dir:
            from rsc_toolkit.datasets.feature_cache import FeatureCache
            processor_config = self._get_processor_config()
            self._feature_cache = FeatureCache(
                cache_dir=pathlib.Path(self.cache_dir),
                processor_config=processor_config,
                processor_name=self.__class__.__name__
            )
        return self
    
    def _get_processor_config(self) -> Dict[str, Any]:
        """Extract config that affects feature computation"""
        # Override in subclasses to return relevant config
        # Exclude cache-related fields and private attributes
        config = self.model_dump(exclude={'cache_features', 'cache_dir', '_feature_cache', 'augmentations'})
        return config
    
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
        if self._feature_cache and self._feature_cache.is_cached(sample_id):
            logger.debug(f"Cache hit for {sample_id}")
            return self._feature_cache.get(sample_id)
        
        # Compute features
        features = self.make_features(audio, sample_id=sample_id)
        
        # Cache if enabled
        if self._feature_cache:
            self._feature_cache.put(sample_id, features)
        
        return features
