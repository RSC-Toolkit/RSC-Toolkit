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
) -> torch.Tensor:
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
        fbank = (fbank - mean) / (std * 2)
    return fbank


class MelBankAudioProcessor(BaseAudioProcessor):

    def __init__(
        self,
        n_mels: int,
        sampling_rate: int,
        target_frames: int = 1024,
        frame_shift: int = 10,
        window_type: str = "hanning",
        dither: float = 0.0,
        normalize: bool = True,
        **kwargs,
    ):
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.target_frames = target_frames
        self.frame_shift = frame_shift
        self.window_type = window_type
        self.dither = dither
        self.normalize = normalize
        super().__init__(**kwargs)
    
    def get_processor_config(self) -> Dict[str, Any]:
        return {
            "sampling_rate": self.sampling_rate,
            "n_mels": self.n_mels,
            "target_frames": self.target_frames,
            "frame_shift": self.frame_shift,
            "window_type": self.window_type,
            "dither": self.dither,
            "normalize": self.normalize,
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
