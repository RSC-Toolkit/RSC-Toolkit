"""
OPERA 배치 멜 전처리 — eval-audio-repr evar/ar_opera.py get_entire_signal_librosa (OPERA src/util.py 인용) 의 torch 재현.

원본(EMBC'25 동결 평가 경로)은 **배치 전체**를 numpy 로 내려 librosa 로 처리한다:
  1) librosa.effects.trim(batch, frame_length=sr/10, hop=sr/20) — 배치 (B,T) 를 다채널로 보고 모든 샘플이 무음(배치 최대 프레임
     파워 대비 -60 dB 미만)인 앞·뒤 프레임만 자른다.
  2) 잘린 길이가 input_sec(8 s) 미만이면 split_pad_sample 로 패딩하려 하지만, 2-D 배열에서는 len()=B 라 원본이 예외로 죽는다
     → 실제로 trim 이 한 프레임이라도 잘리면 원본 실행이 불가능하다. 여기서는 같은 상황에서 RuntimeError 를 낸다.
  3) librosa.feature.melspectrogram(n_mels 64, n_fft 1024, hop 512, 50–8000 Hz, power 2, center, pad 'constant')
     → power_to_db(ref=np.max: **배치** 최대, top_db 80) → **배치** 전체 min-max → (B, 64, T).
  배치 구성에 따라 값이 달라지는 quirk 이며(배치 의존 정규화), 그대로 유지한다.

멜 계산은 GPU(torch.stft + librosa.filters.mel 필터뱅크)로 하고 librosa 결과와 골든으로 대조한다 (tests/golden/opera).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def librosa_trim_indices(y: torch.Tensor, frame_length: int, hop_length: int, top_db: float = 60.0) -> Tuple[int, int]:
    """librosa.effects.trim(y (B,T) or (T,), frame_length, hop_length) 의 (start, end) 샘플 인덱스를 torch 로 계산."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    T = y.shape[-1]
    pad = frame_length // 2
    yp = torch.nn.functional.pad(y, (pad, pad))  # librosa.feature.rms center=True, pad_mode='constant'
    frames = yp.unfold(-1, frame_length, hop_length)  # (B, n_frames, frame_length)
    mse = (frames.abs() ** 2).mean(-1)  # rms**2
    amin = 1e-10
    ref = mse.max()
    db = 10.0 * torch.log10(mse.clamp_min(amin)) - 10.0 * torch.log10(ref.clamp_min(amin))
    db = db.amax(dim=0)  # aggregate=np.max over non-time axes
    nonsilent = torch.nonzero(db > -top_db).flatten()
    if nonsilent.numel() == 0:
        return 0, 0
    start = int(nonsilent[0].item()) * hop_length
    end = min(T, (int(nonsilent[-1].item()) + 1) * hop_length)
    return start, end


class OperaBatchMel(nn.Module):
    """(B, T) 16 kHz 파형 배치 → (B, n_mels, frames) 배치 정규화 멜 (위 설명의 3 단계)."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64, f_min: float = 50.0, f_max: float = 8000.0,
                 n_fft: int = 1024, hop_length: int = 512, input_sec: float = 8.0, top_db: float = 80.0):
        super().__init__()
        import librosa

        self.sample_rate, self.n_fft, self.hop_length = int(sample_rate), int(n_fft), int(hop_length)
        self.input_sec, self.top_db = float(input_sec), float(top_db)
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True), persistent=False)
        fb = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)  # (n_mels, 513)
        self.register_buffer("mel_fb", torch.from_numpy(np.asarray(fb, dtype=np.float32)), persistent=False)

    @torch.no_grad()
    def trim(self, wave: torch.Tensor) -> torch.Tensor:
        frame_len = int(self.sample_rate / 10)
        start, end = librosa_trim_indices(wave, frame_len, int(frame_len / 2))
        if (start, end) != (0, wave.shape[-1]):
            if (end - start) / self.sample_rate < self.input_sec:
                raise RuntimeError(
                    f"배치 단위 trim 이 [{start}:{end}] 로 잘려 {self.input_sec}s 미만이 됐다 — 원본(ar_opera.get_entire_signal_librosa) 은 "
                    "이 경우 2-D 배열 패딩에서 예외로 죽으므로 재현 대상 밖이다. 배치 구성(무음 가장자리)을 확인할 것.")
            wave = wave[..., start:end]
        return wave

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 3:
            wave = wave.squeeze(1)
        wave = self.trim(wave.float())
        spec = torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window,
                          center=True, pad_mode="constant", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2  # (B, 513, frames)
        S = torch.matmul(self.mel_fb, power)  # (B, n_mels, frames)
        amin = 1e-10
        log_spec = 10.0 * torch.log10(S.clamp_min(amin)) - 10.0 * torch.log10(S.max().clamp_min(amin))
        log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)
        lo, hi = log_spec.min(), log_spec.max()
        if hi != lo:
            log_spec = (log_spec - lo) / (hi - lo)
        return log_spec
