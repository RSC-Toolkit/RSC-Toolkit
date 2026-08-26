"""
파형 밴드패스 필터 — Lungmix(ICASSP 2025) §III-B '50 Hz–1500 Hz 밴드패스' 의 **논문 명세 재구현**.

논문은 필터 종류·차수·zero-phase 여부를 적지 않았고 공식 코드에도 없다(AUTHOR_QUESTIONS LungMix #6).
가정(저자 확인 전, 전부 플래그): Butterworth 4차, zero-phase(scipy.signal.sosfiltfilt; 유효 8차 크기 응답), 믹싱 전 각 파형에 적용.
scipy 가 없으면 torchaudio 의 biquad 계열로 대체하지 않고 ImportError 를 낸다(수치 정의가 달라지므로).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


class BandpassFilter:
    def __init__(self, sampling_rate: int, low_hz: float = 50.0, high_hz: float = 1500.0, order: int = 4,
                 zero_phase: bool = True, kind: str = "butter"):
        from scipy import signal  # .venv 에 있음 (1.17)

        if kind != "butter":
            raise ValueError("kind 는 butter 만 지원한다 (논문 미기재 → Butterworth 가정)")
        nyq = 0.5 * float(sampling_rate)
        if not (0 < low_hz < high_hz < nyq):
            raise ValueError(f"잘못된 밴드: {low_hz}-{high_hz} Hz (nyquist {nyq})")
        self.sampling_rate, self.low_hz, self.high_hz = int(sampling_rate), float(low_hz), float(high_hz)
        self.order, self.zero_phase, self.kind = int(order), bool(zero_phase), kind
        self.sos = signal.butter(self.order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
        self._signal = signal

    def config(self) -> Dict[str, Any]:
        return {"kind": self.kind, "low_hz": self.low_hz, "high_hz": self.high_hz, "order": self.order,
                "zero_phase": self.zero_phase, "sampling_rate": self.sampling_rate}

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (C, L) 또는 (L,) → 같은 모양 float32. 마지막 축을 따라 필터링."""
        x = audio.detach().cpu().numpy().astype(np.float64)
        if self.zero_phase:
            y = self._signal.sosfiltfilt(self.sos, x, axis=-1)
        else:
            y = self._signal.sosfilt(self.sos, x, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(y).astype(np.float32))

    def frequency_response(self, freqs_hz, worN: int = 8192):
        """|H(f)| (zero-phase 면 |H|^2). 테스트용."""
        w, h = self._signal.sosfreqz(self.sos, worN=worN, fs=self.sampling_rate)
        mag = np.abs(h) ** (2 if self.zero_phase else 1)
        return np.interp(np.asarray(freqs_hz, dtype=float), w, mag)
