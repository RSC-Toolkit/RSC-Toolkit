from typing import Optional, Dict, Any, ClassVar, Mapping
from abc import ABC, abstractmethod
import pathlib
import logging

import torch

from rsc_toolkit.datasets.feature_cache import FeatureCache


logger = logging.getLogger(__name__)


class BaseAudioProcessor(ABC):
    """피처 프런트엔드 계약.

    registry(`rsc_toolkit.registries.PROCESSORS`, `data.params.features.kind`)로 조회되는 프로세서는
    두 가지를 더 선언한다:

    - `is_spectrogram`: True 면 datamodule 이 학습 배치에 SpecAugment 를 걸 수 있다
      (내장은 mel/fbank 만 True — 예전 `{"mel","fbank"}` 화이트리스트와 같은 진리표).
    - `from_options(opts, *, sampling_rate, feature_cache)`: `features` config(dict) 로부터 자신을
      만든다. 어떤 키를 읽을지는 각 kind 가 안다. 새 kind 의 전용 파라미터는 `features.params` 에 넣는다.

    `get_processor_config()` 는 **피처 캐시 키**다 — 생성 경로가 바뀌어도 이 dict 가 같으면 캐시가
    그대로 적중한다. 손대지 말 것.
    """

    is_spectrogram: ClassVar[bool] = False

    def __init__(
        self,
        feature_cache: Optional[FeatureCache] = None

    ) -> None:
        self.feature_cache = feature_cache

    @classmethod
    def from_options(cls, opts: Mapping[str, Any], *, sampling_rate: int, feature_cache: Optional[FeatureCache]):
        """`features` config 로부터 프로세서를 만든다. 내장 kind 는 각자 구현; 기본은 미구현 오류."""
        raise NotImplementedError(
            f"{cls.__name__} 은 from_options 를 구현하지 않았다 — features.kind 로 쓰려면 구현하라"
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
