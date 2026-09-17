"""
A 'feature' processor that passes the waveform through — for backbones that compute fbank inside the model, such as BEATs.

Uses the IndexDataModule feature path (IndexFeatureDataset) as is, but does no computation or caching
(with feature_cache=None, make_features_cached skips the cache).
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
        # Backbones that compute fbank inside the model (BEATs, etc.): raw waveform, no cache — feature_cache is ignored
        return cls(sampling_rate=sampling_rate, feature_cache=None)

    def get_processor_config(self) -> Dict[str, Any]:
        return {"kind": "waveform", "sampling_rate": self.sampling_rate}

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        return audio.float()  # (L,)
