"""
RespireNet(EMBC 2021)의 스펙트로그램 이미지 특징 — 공식 코드 충실 이식 (P0).

출처: https://github.com/microsoft/RespireNet utils.py create_mel_raw + image_dataloader.py __getitem__.

체인 (원본 그대로, uint8 비트 일치 목표):
  4 kHz 8초 파형 → librosa mel(64, n_fft 256, hop 128, 50–2000 Hz, power 2)
  → power_to_db(ref=max) → 이미지별 min-max → ×255 uint8 → magma 컬러맵(cv2 LUT, BGR)
  → cv2.resize ×3 (INTER_LINEAR) → cv2.flip(0) → BGR→RGB
  → BRC(blank region clipping): 1배 magma 이미지를 GRAY 로 바꿔 <10 을 0 으로, 위(고주파)부터
    검은 픽셀 비율 <0.8 인 첫 행 row 까지의 (row+1)*3 행을 3배 이미지에서 잘라내고
    (753, 192) 로 재 resize. train/test 공통 전처리.
  → ToTensor(/255) + Normalize(mean, std)   (eval.py:63 하드코딩 상수가 기본값)

magma LUT 는 cmapy.cmap('magma') 를 저장한 것(processors/data/magma_lut_bgr.npy) — cmapy/matplotlib
의존을 피하고 cv2.applyColorMap(userColor) 와 같은 결과.
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

# eval.py:63 — 저자 학습 run 의 train mean/std (train.py 는 매 run 재계산)
DEFAULT_MEAN = (0.5091, 0.1739, 0.4363)
DEFAULT_STD = (0.2288, 0.1285, 0.0743)


def create_mel_raw(current_window: np.ndarray, sample_rate: int, n_mels: int = 128, f_min: int = 50,
                   f_max: int = 4000, nfft: int = 2048, hop: int = 512, resz: int = 1) -> np.ndarray:
    """원본 utils.py:237-247 그대로 (BGR uint8 HWC)."""
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
    """원본 image_dataloader.py:252-272 — RGB uint8 (n_mels*3, W, 3), BRC 적용."""
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
    """torchvision ToTensor + Normalize 와 동일 (HWC uint8 → CHW float)."""
    x = torch.from_numpy(np.ascontiguousarray(img_rgb_uint8)).permute(2, 0, 1).float().div(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - m) / s


class RespireNetImageProcessor(BaseAudioProcessor):
    """파형(4 kHz, 8초) → 정규화된 (3, 192, 753) 텐서."""

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
