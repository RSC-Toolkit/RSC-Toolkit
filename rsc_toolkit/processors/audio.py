from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class AudioPreprocessConfig:
    desired_length_sec: Optional[float] = None
    # "repeat"      : repeat padding (no fade)
    # "zero"        : zero padding at the end
    # "repeat_fade" : same as cut_pad_sample_torchaudio of the patch-mix/SCL family papers —
    #                 repeat padding, then a linear fade_out over the last sr/16 samples. Original quirk kept:
    #                 even when the length equals the target exactly, the pad branch runs and fade_out is applied.
    #                 (ref: raymin0223/patch-mix util/icbhi_util.py cut_pad_sample_torchaudio)
    # "sclmeta"     : ilyassmoummad/scl_icbhi2017 dataset.py — applies fade in/out (sr/16) right after loading the segment,
    #                 then truncates the beginning if too long, or cyclically repeats + truncates and re-applies fade_out (sr/16) if too short.
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

        if audio.ndim != 2:
            raise ValueError(f"AudioPreprocessor expects a 2-D (channels, samples) tensor: {tuple(audio.shape)}")
        channels, length = audio.shape
        if length <= 0:
            # A zero-length segment crashes repeat padding with ZeroDivisionError / "negative dimension" (repeat, repeat_fade)
            # and silently yields silence with zero padding. Index defects must surface here.
            raise ValueError("cannot pad a zero-length audio segment — check start/end in the index (t0 ≥ t1)")

        if self.cfg.pad_type == "sclmeta":
            import torchaudio.transforms as T

            fade_samples = int(self.sampling_rate / 16)
            if length < fade_samples:
                # The original code (scl_icbhi2017) also fades the raw segment first, so torchaudio Fade crashes with
                # "negative dimension" on segments shorter than sr/16 — raise an error that names the cause (the current index minimum is 0.126 s, so it does not occur).
                raise ValueError(
                    f"sclmeta padding: segment length of {length} samples is shorter than the fade length sr/16={fade_samples} — check the index segment"
                )
            audio = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear")(audio)
            if audio.shape[-1] > target_samples:
                return audio[..., :target_samples]
            ratio = -(-target_samples // audio.shape[-1])
            audio = audio.repeat(1, ratio)[..., :target_samples]
            return T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape="linear")(audio)

        if self.cfg.pad_type == "repeat_fade":
            # Same control flow as the original code: if too long, keep the beginning (no fade); otherwise
            # (even at equal length) take the repeat-padding branch and apply fade_out.
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
