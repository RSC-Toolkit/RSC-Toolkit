# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Optional, Dict, Any

import torch
import torchaudio

from .base import BaseAudioProcessor


def _ensure_2d_audio(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.dim() == 2:
        return audio
    raise ValueError(f"Expected 1D or 2D audio, got shape {tuple(audio.shape)}")


def generate_fbank(
    audio: torch.Tensor,
    sampling_rate: int,
    n_mels: int = 128,
    frame_shift: int = 10,
    window_type: str = "hanning",
    dither: float = 0.0,
    normalize: bool = True,
    mean: float = -4.2677393,
    std: float = 4.5689974,
    per_sample_stats: bool = False,
) -> torch.Tensor:
    """
    per_sample_stats=True reproduces the non-AST branch of the patch-mix/SG-SCL family official code:
    normalizes (x - mean) / (std * 2) with the fbank's own mean/std instead of fixed constants.
    (ref: kaen2891/sgscl util/icbhi_util.py generate_fbank — only AST uses fixed constants)
    """
    audio = _ensure_2d_audio(audio)
    fbank = torchaudio.compliance.kaldi.fbank(
        audio,
        htk_compat=True,
        sample_frequency=sampling_rate,
        use_energy=False,
        window_type=window_type,
        num_mel_bins=n_mels,
        dither=dither,
        frame_shift=frame_shift,
    )
    if normalize:
        if per_sample_stats:
            mean, std = fbank.mean(), fbank.std()
        fbank = (fbank - mean) / (std * 2)
    return fbank


class MelBankAudioProcessor(BaseAudioProcessor):
    is_spectrogram = True  # kind mel/fbank — SpecAugment allowed on training batches

    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):
        # constructor arguments for kind in {"mel","fbank"}
        return cls(
            sampling_rate=sampling_rate,
            n_mels=opts["n_mels"],
            target_frames=opts["target_frames"],
            frame_shift=opts["frame_shift"],
            window_type=opts["window_type"],
            dither=opts["dither"],
            normalize=opts["normalize"],
            per_sample_stats=opts["per_sample_stats"],
            feature_cache=feature_cache,
        )

    def __init__(
        self,
        n_mels: int,
        sampling_rate: int,
        target_frames: int = 1024,
        frame_shift: int = 10,
        window_type: str = "hanning",
        dither: float = 0.0,
        normalize: bool = True,
        per_sample_stats: bool = False,
        **kwargs,
    ):
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.target_frames = target_frames
        self.frame_shift = frame_shift
        self.window_type = window_type
        self.dither = dither
        self.normalize = normalize
        self.per_sample_stats = per_sample_stats
        super().__init__(**kwargs)
    
    def get_processor_config(self) -> Dict[str, Any]:
        # per_sample_stats must be part of the cache key — features with different normalization must not mix
        return {
            "sampling_rate": self.sampling_rate,
            "n_mels": self.n_mels,
            "target_frames": self.target_frames,
            "frame_shift": self.frame_shift,
            "window_type": self.window_type,
            "dither": self.dither,
            "normalize": self.normalize,
            "per_sample_stats": self.per_sample_stats,
        }

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        fbank = generate_fbank(
            audio,
            sampling_rate=self.sampling_rate,
            n_mels=self.n_mels,
            frame_shift=self.frame_shift,
            window_type=self.window_type,
            dither=self.dither,
            normalize=self.normalize,
            per_sample_stats=self.per_sample_stats,
        )
        # fbank: [frames, n_mels] -> [n_mels, frames]
        fbank = fbank.transpose(0, 1)

        if self.target_frames is not None:
            frames = fbank.shape[1]
            if frames < self.target_frames:
                pad = self.target_frames - frames
                fbank = torch.nn.functional.pad(fbank, (0, pad))
            elif frames > self.target_frames:
                fbank = fbank[:, : self.target_frames]

        return fbank
