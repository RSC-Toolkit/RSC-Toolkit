# Portions of this file are ported from scl_icbhi2017 (https://github.com/ilyassmoummad/scl_icbhi2017), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Spectrogram of SCL-metadata (Moummad & Farrugia, ICASSP 2023 / arXiv 2210.16192) — faithful port of the official code.

Source: https://github.com/ilyassmoummad/scl_icbhi2017 main.py:29-34,59-62 + utils.py.
Per-sample steps (cacheable): torchaudio MelSpectrogram(sample_rate 16000 (default), n_fft 1024, win 1024, hop 512,
n_mels 64, f_min 50, f_max 2000, power 2) → AmplitudeToDB(power, top_db None) → (1, 64, 251) for 8 s.
Batch steps (inside the model, batch-dependent): min-max over the whole batch → [SpecAugment] → Standardize(0.3690, 0.0255).
"""
from typing import Any, Dict, Optional

import torch
import torchaudio.transforms as T

from rsc_toolkit.processors.base import BaseAudioProcessor

STANDARDIZE_MEAN = 0.3690  # utils.py Standardize defaults ("official split mean & std")
STANDARDIZE_STD = 0.0255


def batch_minmax(x: torch.Tensor) -> torch.Tensor:
    """Second half of utils.py Normalize.forward: scale to [0,1] with the min/max of the whole batch tensor."""
    return (x - x.min()) / (x.max() - x.min())


def standardize(x: torch.Tensor, mean: float = STANDARDIZE_MEAN, std: float = STANDARDIZE_STD) -> torch.Tensor:
    return (x - mean) / std


class SCLMetaMelProcessor(BaseAudioProcessor):
    def __init__(self, sampling_rate: int = 16000, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512,
                 n_mels: int = 64, f_min: float = 50, f_max: float = 2000, **kwargs):
        self.sampling_rate = int(sampling_rate)
        self.cfg = dict(n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max)
        # The original code does not pass sample_rate, so the torchaudio default 16000 is used — set the same value explicitly
        self.melspec = T.MelSpectrogram(sample_rate=self.sampling_rate, **self.cfg)
        self.power_to_db = T.AmplitudeToDB()
        super().__init__(**kwargs)

    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):
        # constructor arguments for kind=="sclmeta_mel"
        return cls(
            sampling_rate=sampling_rate, n_mels=opts["n_mels"], n_fft=opts["n_fft"],
            win_length=opts["win_length"] or opts["n_fft"], hop_length=opts["hop_length"],
            f_min=opts["f_min"], f_max=opts["f_max"], feature_cache=feature_cache,
        )

    def get_processor_config(self) -> Dict[str, Any]:
        return {"kind": "sclmeta_mel", "sampling_rate": self.sampling_rate, **self.cfg}

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        with torch.no_grad():
            return self.power_to_db(self.melspec(audio.float()))  # (1, n_mels, frames)
