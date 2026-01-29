from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class AudioPreprocessConfig:
    desired_length_sec: Optional[float] = None
    pad_type: str = "repeat"  # "repeat" | "zero"


class AudioPreprocessor:
    def __init__(self, sampling_rate: int, cfg: AudioPreprocessConfig):
        self.sampling_rate = int(sampling_rate)
        self.cfg = cfg

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.cfg.desired_length_sec is None:
            return audio

        target_samples = int(self.cfg.desired_length_sec * self.sampling_rate)
        if target_samples <= 0:
            return audio

        channels, length = audio.shape
        if length == target_samples:
            return audio
        if length > target_samples:
            return audio[:, :target_samples]

        pad = target_samples - length
        if self.cfg.pad_type == "zero":
            return torch.nn.functional.pad(audio, (0, pad))
        if self.cfg.pad_type == "repeat":
            reps = (target_samples + length - 1) // length
            expanded = audio.repeat(1, reps)
            return expanded[:, :target_samples]

        raise ValueError(f"Unknown pad_type: {self.cfg.pad_type}")
