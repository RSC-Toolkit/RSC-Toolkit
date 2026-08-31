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
        # transformers 의 ClapFeatureExtractor 는 **torch 1-D 텐서를 2-D(배치)로 오인**해
        # 샘플 수 × 샘플 수 크기의 배열을 만들려다 MemoryError 를 낸다(48 kHz·8 s 면 3 TiB 요구).
        # numpy 1-D 로 넘기면 정상 경로를 탄다. 이 우회는 SPRSound OOD 경로(datasets/sprsound_ood.py)
        # 에서 먼저 쓰였고, 그 결과가 BTS 저자 배포 텐서와 비트 단위로 일치함을 ICBHI 에서 확인했다.
        feats = self.processor(audio=audio.detach().cpu().numpy(), return_tensors="pt", padding=True)["input_features"]
        return feats
    