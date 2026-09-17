"""
OPERA batch mel preprocessing — torch re-implementation of get_entire_signal_librosa in eval-audio-repr evar/ar_opera.py (quoting OPERA src/util.py).

The original code (EMBC'25 frozen evaluation path) moves the **whole batch** to numpy and processes it with librosa:
  1) librosa.effects.trim(batch, frame_length=sr/10, hop=sr/20) — treats the batch (B,T) as multichannel and trims only leading/trailing
     frames where every sample is silent (below -60 dB relative to the batch's maximum frame power).
  2) If the trimmed length is below input_sec (8 s), it tries to pad with split_pad_sample, but on a 2-D array len()=B, so the original crashes
     → in practice the original cannot run if trim removes even one frame. Here the same situation raises RuntimeError.
  3) librosa.feature.melspectrogram(n_mels 64, n_fft 1024, hop 512, 50–8000 Hz, power 2, center, pad 'constant')
     → power_to_db(ref=np.max: **batch** maximum, top_db 80) → min-max over the whole **batch** → (B, 64, T).
  Values depend on the batch composition (batch-dependent normalization); this quirk is kept as is.

The mel computation runs on the GPU (torch.stft + librosa.filters.mel filterbank) and is compared against librosa goldens (tests/golden/opera).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def librosa_trim_indices(y: torch.Tensor, frame_length: int, hop_length: int, top_db: float = 60.0) -> Tuple[int, int]:
    """Compute the (start, end) sample indices of librosa.effects.trim(y (B,T) or (T,), frame_length, hop_length) in torch."""
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
    """(B, T) 16 kHz waveform batch → (B, n_mels, frames) batch-normalized mel (the 3 steps described above)."""

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
                    f"batch-level trim cut the batch to [{start}:{end}], shorter than {self.input_sec}s — the original code (ar_opera.get_entire_signal_librosa) "
                    "crashes on 2-D array padding in this case, so it is outside the reproduction scope. Check the batch composition (silent edges).")
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
