"""
파형을 그대로 넘기는 '피처' 프로세서 — BEATs 처럼 모델 내부에서 fbank 를 만드는 백본용.

IndexDataModule 의 피처 경로(IndexFeatureDataset)를 그대로 타되 계산·캐시는 하지 않는다
(feature_cache=None 으로 두면 make_features_cached 가 캐시를 건너뛴다).
"""
from typing import Any, Dict, Optional

import torch

from rsc_toolkit.processors.base import BaseAudioProcessor


class WaveformProcessor(BaseAudioProcessor):
    def __init__(self, sampling_rate: int, **kwargs):
        self.sampling_rate = int(sampling_rate)
        super().__init__(**kwargs)

    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):
        # 모델 내부에서 fbank 를 만드는 백본(BEATs 등): 파형 그대로, 캐시 없음 — feature_cache 는 무시한다
        return cls(sampling_rate=sampling_rate, feature_cache=None)

    def get_processor_config(self) -> Dict[str, Any]:
        return {"kind": "waveform", "sampling_rate": self.sampling_rate}

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        return audio.float()  # (L,)
