from typing import Optional, Dict, Any
import logging
import functools

import torch
from transformers import ClapProcessor

from rsc_toolkit.processors.base import BaseAudioProcessor


logger = logging.getLogger(__name__)


class CLAPProcessor(BaseAudioProcessor):
    """
    CLAP (Contrastive Language-Audio Pre-training) feature extractor.
    This is slow, so caching is highly recommended.
    
    Requires: laion_clap package
    Install with: pip install laion-clap
    """

    def __init__(
        self,
        sampling_rate: int,
        model_name: str = "laion/clap-htsat-unfused",
        **kwargs
    ):
        self.sampling_rate = sampling_rate
        self.model_name = model_name
        self.processor = ClapProcessor.from_pretrained(
            self.model_name,
            sampling_rate=self.sampling_rate
        )
        super().__init__(**kwargs)
    
    def get_processor_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
        }
    
    def make_features(
        self, 
        audio: torch.Tensor, 
        sample_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract CLAP features from audio.
        Audio should be shape [channels, samples] or [samples]
        """
        assert len(audio.shape) == 1, f"Got unexpected shape: {audio.shape}"
        return self.processor(audio=audio, return_tensors="pt", padding=True)['input_features']
    