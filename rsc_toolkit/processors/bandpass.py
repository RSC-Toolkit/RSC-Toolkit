"""
Waveform band-pass filter — **re-implementation from the paper description** of the '50 Hz–1500 Hz band-pass' in Lungmix (ICASSP 2025) §III-B.

The paper does not specify the filter type, order, or whether it is zero-phase, and the official code does not include it.
Assumptions (not specified in the paper; all configurable): 4th-order Butterworth, zero-phase (scipy.signal.sosfiltfilt; effective 8th-order magnitude response), applied to each waveform before mixing.
Without scipy an ImportError is raised instead of falling back to torchaudio biquads (the numerical definition would differ).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


class BandpassFilter:
    def __init__(self, sampling_rate: int, low_hz: float = 50.0, high_hz: float = 1500.0, order: int = 4,
                 zero_phase: bool = True, kind: str = "butter"):
        from scipy import signal  # present in .venv (1.17)

        if kind != "butter":
            raise ValueError("only kind=butter is supported (not specified in the paper → Butterworth assumed)")
        nyq = 0.5 * float(sampling_rate)
        if not (0 < low_hz < high_hz < nyq):
            raise ValueError(f"invalid band: {low_hz}-{high_hz} Hz (nyquist {nyq})")
        self.sampling_rate, self.low_hz, self.high_hz = int(sampling_rate), float(low_hz), float(high_hz)
        self.order, self.zero_phase, self.kind = int(order), bool(zero_phase), kind
        self.sos = signal.butter(self.order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
        self._signal = signal

    def config(self) -> Dict[str, Any]:
        return {"kind": self.kind, "low_hz": self.low_hz, "high_hz": self.high_hz, "order": self.order,
                "zero_phase": self.zero_phase, "sampling_rate": self.sampling_rate}

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (C, L) or (L,) → float32 of the same shape. Filters along the last axis."""
        x = audio.detach().cpu().numpy().astype(np.float64)
        if self.zero_phase:
            y = self._signal.sosfiltfilt(self.sos, x, axis=-1)
        else:
            y = self._signal.sosfilt(self.sos, x, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(y).astype(np.float32))

    def frequency_response(self, freqs_hz, worN: int = 8192):
        """|H(f)| (|H|^2 if zero-phase). For tests."""
        w, h = self._signal.sosfreqz(self.sos, worN=worN, fs=self.sampling_rate)
        mag = np.abs(h) ** (2 if self.zero_phase else 1)
        return np.interp(np.asarray(freqs_hz, dtype=float), w, mag)
