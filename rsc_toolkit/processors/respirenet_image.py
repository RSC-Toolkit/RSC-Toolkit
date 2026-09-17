"""
Spectrogram image features of RespireNet (EMBC 2021) — faithful port of the official code (exact port, verified against goldens).

Source: https://github.com/microsoft/RespireNet utils.py create_mel_raw + image_dataloader.py __getitem__.

Preprocessing chain (as in the original code, targeting bit-identical uint8):
  4 kHz 8 s waveform → librosa mel(64, n_fft 256, hop 128, 50–2000 Hz, power 2)
  → power_to_db(ref=max) → per-image min-max → ×255 uint8 → magma colormap (cv2 LUT, BGR)
  → cv2.resize ×3 (INTER_LINEAR) → cv2.flip(0) → BGR→RGB
  → BRC (blank region clipping): convert the 1× magma image to GRAY, set <10 to 0, find from the top (high frequency)
    the first row whose black-pixel ratio is <0.8, cut (row+1)*3 rows from the 3× image,
    and resize back to (753, 192). Shared by train/test.
  → ToTensor(/255) + Normalize(mean, std)   (defaults are the constants hard-coded in eval.py:63)

The magma LUT is a saved cmapy.cmap('magma') (processors/data/magma_lut_bgr.npy) — avoids the cmapy/matplotlib
dependency and gives the same result as cv2.applyColorMap(userColor).
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional, Sequence

import cv2
import librosa
import numpy as np
import torch

from rsc_toolkit.processors.base import BaseAudioProcessor

_LUT_PATH = pathlib.Path(__file__).with_name("data") / "magma_lut_bgr.npy"
MAGMA_LUT_BGR = np.load(_LUT_PATH)  # (256, 1, 3) uint8

# eval.py:63 — train mean/std of the authors' training run (train.py recomputes it per run)
DEFAULT_MEAN = (0.5091, 0.1739, 0.4363)
DEFAULT_STD = (0.2288, 0.1285, 0.0743)


def create_mel_raw(current_window: np.ndarray, sample_rate: int, n_mels: int = 128, f_min: int = 50,
                   f_max: int = 4000, nfft: int = 2048, hop: int = 512, resz: int = 1) -> np.ndarray:
    """Same as original utils.py:237-247 (BGR uint8 HWC)."""
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min,
                                       fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), MAGMA_LUT_BGR)
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width * resz, height * resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img


def respirenet_image(audio: np.ndarray, sample_rate: int = 4000, n_mels: int = 64, nfft: int = 256,
                     hop: int = 128, f_max: int = 2000) -> np.ndarray:
    """Original image_dataloader.py:252-272 — RGB uint8 (n_mels*3, W, 3), with BRC."""
    audio_image = cv2.cvtColor(create_mel_raw(audio, sample_rate, f_max=f_max, n_mels=n_mels,
                                              nfft=nfft, hop=hop, resz=3), cv2.COLOR_BGR2RGB)
    audio_raw_gray = cv2.cvtColor(create_mel_raw(audio, sample_rate, f_max=f_max, n_mels=n_mels,
                                                 nfft=nfft, hop=hop), cv2.COLOR_BGR2GRAY)
    audio_raw_gray[audio_raw_gray < 10] = 0
    for row in range(audio_raw_gray.shape[0]):
        black_percent = len(np.where(audio_raw_gray[row, :] == 0)[0]) / len(audio_raw_gray[row, :])
        if black_percent < 0.80:
            break
    if (row + 1) * 3 < audio_image.shape[0]:
        audio_image = audio_image[(row + 1) * 3:, :, :]
    audio_image = cv2.resize(audio_image, (audio_image.shape[1], n_mels * 3), interpolation=cv2.INTER_LINEAR)
    return audio_image


def normalize_image(img_rgb_uint8: np.ndarray, mean: Sequence[float] = DEFAULT_MEAN,
                    std: Sequence[float] = DEFAULT_STD) -> torch.Tensor:
    """Same as torchvision ToTensor + Normalize (HWC uint8 → CHW float)."""
    x = torch.from_numpy(np.ascontiguousarray(img_rgb_uint8)).permute(2, 0, 1).float().div(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - m) / s


class RespireNetImageProcessor(BaseAudioProcessor):
    """Waveform (4 kHz, 8 s) → normalized (3, 192, 753) tensor."""

    def __init__(self, sampling_rate: int = 4000, n_mels: int = 64, nfft: int = 256, hop: int = 128,
                 f_max: int = 2000, mean: Sequence[float] = DEFAULT_MEAN, std: Sequence[float] = DEFAULT_STD,
                 **kwargs):
        self.sampling_rate, self.n_mels, self.nfft, self.hop, self.f_max = sampling_rate, n_mels, nfft, hop, f_max
        self.mean, self.std = tuple(mean), tuple(std)
        super().__init__(**kwargs)

    def get_processor_config(self) -> Dict[str, Any]:
        return {"kind": "respirenet_image", "sampling_rate": self.sampling_rate, "n_mels": self.n_mels,
                "nfft": self.nfft, "hop": self.hop, "f_max": self.f_max, "mean": self.mean, "std": self.std}

    def image(self, audio) -> np.ndarray:
        audio = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        return respirenet_image(audio.astype(np.float32), self.sampling_rate, self.n_mels, self.nfft, self.hop, self.f_max)

    def make_features(self, audio, sample_id: Optional[str] = None) -> torch.Tensor:
        return normalize_image(self.image(audio), self.mean, self.std)
