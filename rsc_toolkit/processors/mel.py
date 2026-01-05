from typing import Optional
from rsc_toolkit.augmentations import Augmentation
from rsc_toolkit.processors.base import BaseAudioProcessor
from rsc_toolkit.processors.util import generate_fbank
import torch


class MelBankAudioProcessor(BaseAudioProcessor):

    n_mels: int
    sample_rate: int
    
    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        fbank = generate_fbank(audio, sample_rate=self.sample_rate, n_mels=self.n_mels)
        return torch.from_numpy(fbank) if not isinstance(fbank, torch.Tensor) else fbank
