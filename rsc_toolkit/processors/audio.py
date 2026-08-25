from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class AudioPreprocessConfig:
    desired_length_sec: Optional[float] = None
    # "repeat"      : 반복 패딩 (fade 없음)
    # "zero"        : 끝쪽 zero 패딩
    # "repeat_fade" : patch-mix/SCL 계열 논문의 cut_pad_sample_torchaudio 와 동일 —
    #                 반복 패딩 후 끝 sr/16 샘플에 linear fade_out. 원본 quirk 그대로:
    #                 길이가 정확히 목표와 같아도 pad 분기로 들어가 fade_out 이 적용된다.
    #                 (참조: raymin0223/patch-mix util/icbhi_util.py cut_pad_sample_torchaudio)
    # "sclmeta"     : ilyassmoummad/scl_icbhi2017 dataset.py — 구간 로드 직후 fade in/out(sr/16) 을
    #                 먼저 걸고, 길면 앞부분 절단, 짧으면 순환 반복 후 절단 + fade_out(sr/16) 재적용.
    pad_type: str = "repeat"  # "repeat" | "zero" | "repeat_fade" | "sclmeta"


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

        if self.cfg.pad_type == "sclmeta":
            import torchaudio.transforms as T

            fade_samples = int(self.sampling_rate / 16)
            audio = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear")(audio)
            if audio.shape[-1] > target_samples:
                return audio[..., :target_samples]
            ratio = -(-target_samples // audio.shape[-1])
            audio = audio.repeat(1, ratio)[..., :target_samples]
            return T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape="linear")(audio)

        if self.cfg.pad_type == "repeat_fade":
            # 원본 제어 흐름 그대로: 길면 앞부분만 자르고(fade 없음), 그 외에는
            # (길이가 같아도) 반복-패딩 분기로 들어가 fade_out 을 적용한다.
            if length > target_samples:
                return audio[:, :target_samples]
            reps = -(-target_samples // length)  # ceil
            data = audio.repeat(1, reps)[:, :target_samples]
            fade_samples = int(self.sampling_rate / 16)
            if fade_samples > 0:
                import torchaudio.transforms as T

                fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape="linear")
                data = fade_out(data)
            return data

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
