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

    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):
        # constructor arguments for kind=="clap"
        return cls(sampling_rate=sampling_rate, model_name=opts["clap_model_name"], feature_cache=feature_cache)

    def get_processor_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sampling_rate": self.sampling_rate,
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
        # transformers' ClapFeatureExtractor **mistakes a torch 1-D tensor for a 2-D batch** and tries to
        # allocate a samples × samples array, raising MemoryError (3 TiB for 48 kHz, 8 s).
        # Passing a numpy 1-D array takes the normal path. The same workaround is used in the SPRSound OOD path
        # (datasets/sprsound_ood.py); its output is bit-identical to the tensors released by the BTS authors on ICBHI.
        feats = self.processor(audio=audio.detach().cpu().numpy(), return_tensors="pt", padding=True)["input_features"]
        return feats
    