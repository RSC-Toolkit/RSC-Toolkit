from typing import Optional, Dict, Any

import torch
import torchaudio

from .base import BaseAudioProcessor


def generate_fbank(audio, sampling_rate, n_mels=128):
    assert sampling_rate == 16000, 'input audio sampling rate must be 16kHz' # Why ?
    fbank = torchaudio.compliance.kaldi.fbank( # NOTE many of these can be moved to the class and used for cache key
        audio,
        htk_compat=True,
        sample_frequency=sampling_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=n_mels,
        dither=0.0,
        frame_shift=10,
    )
    mean, std = -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2)
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank


class MelBankAudioProcessor(BaseAudioProcessor):

    def __init__(
        self,
        n_mels: int,
        sampling_rate: int,
        **kwargs,
    ):
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        super().__init__(**kwargs)
    
    def get_processor_config(self) -> Dict[str, Any]:
        return {
            "sampling_rate": self.sampling_rate,
            "n_mels": self.n_mels,
        }

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        raise NotImplementedError("Batching and padding still a mystery")
        fbank = generate_fbank(audio, sampling_rate=self.sampling_rate, n_mels=self.n_mels)
        return torch.from_numpy(fbank) if not isinstance(fbank, torch.Tensor) else fbank
