# Portions of this file are adapted from RespireNet (https://github.com/microsoft/RespireNet); that code is under the MIT license.
# Portions of this file are adapted from nlpaug 0.0.14 (https://github.com/makcedward/nlpaug); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
RespireNet training waveform augmentation: a faithful port of the five nlpaug 0.0.14 audio augmentations used by gen_augmented in the original utils.py.

Source: microsoft/RespireNet utils.py:184-197 + nlpaug 0.0.14 nlpaug/model/audio/{audio,noise,speed,loudness,vtlp,pitch}.py.
Original call: one of NoiseAug(), SpeedAug(), LoudnessAug(factor=(0.5,2)), VtlpAug(sr, zone=(0,1)), PitchAug(sr, factor=(-1,3))
is picked with random.randint and augment() (= model.manipulate) is applied once.

The actual behavior of nlpaug 0.0.14 is ported as-is (the behavior, not the intent, is what produced the paper's numbers):
- zone default (0.2, 0.8), coverage 1 → only the middle 60% is augmented (only Vtlp uses zone (0,1), coverage 0.1).
- Speed: {0.5..1.9}\\{1.0} in 0.1 steps, Pitch: randint(-1,3) → {-1,0,1,2} semitones, Loudness: uniform(0.5,2).
- Noise: white noise irfft(randn) added as-is (no scaling), segment position padded via randint.
- Vtlp: includes the nlpaug bug that misreads the librosa.stft output (freq, frames) as (time, freq) and **warps the frame axis**.
- The original nlpaug raises TypeError with librosa ≥0.10 because of positional time_stretch/pitch_shift arguments; it worked in the
  authors' environment (librosa 0.8), so the same operations are called with keyword arguments.
RNG: np.random → the given RandomState, random → the given random.Random (reproducibility).
"""
from __future__ import annotations

import random

import librosa
import numpy as np


class _Audio:
    def __init__(self, zone=(0.2, 0.8), coverage=1.0, factor=None, duration=None, sampling_rate=None):
        self.zone, self.coverage, self.factor, self.duration, self.sampling_rate = zone, coverage, factor, duration, sampling_rate

    @staticmethod
    def pad(data, noise, rng):
        if len(data) - len(noise) == 0:
            start_pos = 0
        else:
            start_pos = rng.randint(0, len(data) - len(noise))
        prefix_padding = np.array([0] * start_pos)
        suffix_padding = np.array([0] * (len(data) - len(noise) - start_pos))
        return np.append(np.append(prefix_padding, noise), suffix_padding)

    def get_augmentation_segment_size(self, data):
        return int(len(data) * (self.zone[1] - self.zone[0]) * self.coverage)

    def get_augment_range_by_coverage(self, data, rng):
        zone_start, zone_end = int(len(data) * self.zone[0]), int(len(data) * self.zone[1])
        zone_size = zone_end - zone_start
        target_size = int(zone_size * self.coverage)
        last_start = zone_start + int(zone_size * (1 - self.coverage))
        if zone_start == last_start:
            start_pos, end_pos = zone_start, zone_end
        else:
            start_pos = rng.randint(zone_start, last_start)
            end_pos = start_pos + target_size
        return start_pos, end_pos


class Noise(_Audio):
    def __init__(self, zone=(0.2, 0.8), coverage=1.0, color="white"):
        super().__init__(zone=zone, coverage=coverage)
        self.color = color

    def color_noise(self, segment_size, rng):
        uneven = segment_size % 2
        fft_size = segment_size // 2 + 1 + uneven
        noise_fft = rng.randn(fft_size)
        # color == 'white': no color weighting
        if uneven:
            noise_fft = noise_fft[:-1]
        return np.fft.irfft(noise_fft)

    def manipulate(self, data, rng):
        noise = self.color_noise(self.get_augmentation_segment_size(data), rng)
        noise = self.pad(data, noise, rng)
        return (data + noise).astype(type(data[0]))


class Speed(_Audio):
    def __init__(self, zone=(0.2, 0.8), coverage=1.0, factor=(0.5, 2)):
        super().__init__(zone=zone, coverage=coverage, factor=factor)

    def get_speed_level(self, rng):
        speeds = [round(i, 1) for i in np.arange(self.factor[0], self.factor[1], 0.1)]
        speeds = [s for s in speeds if s != 1.0]
        return speeds[rng.randint(len(speeds))]

    def manipulate(self, data, rng):
        speed = self.get_speed_level(rng)
        start_pos, end_pos = self.get_augment_range_by_coverage(data, rng)
        aug_data = librosa.effects.time_stretch(data[start_pos:end_pos], rate=speed)
        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0)


class Loudness(_Audio):
    def __init__(self, zone=(0.2, 0.8), coverage=1.0, factor=(0.5, 2)):
        super().__init__(zone=zone, coverage=coverage, factor=factor)

    def manipulate(self, data, rng):
        loudness_level = rng.uniform(self.factor[0], self.factor[1])
        start_pos, end_pos = self.get_augment_range_by_coverage(data, rng)
        aug_data = data[start_pos:end_pos] * loudness_level
        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0)


class Vtlp(_Audio):
    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=0.1, factor=(0.9, 1.1), fhi=4800):
        super().__init__(zone=zone, coverage=coverage, factor=factor, sampling_rate=sampling_rate)
        self.fhi = fhi

    @classmethod
    def get_scale_factors(cls, freq_dim, sampling_rate, fhi=4800, alpha=0.9):
        factors = []
        freqs = np.linspace(0, 1, freq_dim)
        scale = fhi * min(alpha, 1)
        f_boundary = scale / alpha
        half_sr = sampling_rate / 2
        for f in freqs:
            f *= sampling_rate
            if f <= f_boundary:
                factors.append(f * alpha)
            else:
                warp_freq = half_sr - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - f)
                factors.append(warp_freq)
        return np.array(factors)

    def _manipulate(self, audio, sampling_rate, factor):
        stft = librosa.stft(audio)
        time_dim, freq_dim = stft.shape  # nlpaug's axis misreading kept as-is (actually freq, frames)
        data_type = type(stft[0][0])
        factors = self.get_scale_factors(freq_dim, sampling_rate, fhi=self.fhi, alpha=factor)
        factors *= (freq_dim - 1) / max(factors)
        new_stft = np.zeros([time_dim, freq_dim], dtype=data_type)
        for i in range(freq_dim):
            if i == 0 or i + 1 >= freq_dim:
                new_stft[:, i] += stft[:, i]
            else:
                warp_up = factors[i] - np.floor(factors[i])
                warp_down = 1 - warp_up
                pos = int(np.floor(factors[i]))
                new_stft[:, pos] += warp_down * stft[:, i]
                new_stft[:, pos + 1] += warp_up * stft[:, i]
        return librosa.istft(new_stft)

    def manipulate(self, data, rng):
        start_pos, end_pos = self.get_augment_range_by_coverage(data, rng)
        factor = rng.uniform(self.factor[0], self.factor[1])
        aug_data = self._manipulate(data[start_pos:end_pos], sampling_rate=self.sampling_rate, factor=factor)
        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0).astype(type(data[0]))


class Pitch(_Audio):
    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=1.0, factor=(-10, 10)):
        super().__init__(zone=zone, coverage=coverage, factor=factor, sampling_rate=sampling_rate)

    def manipulate(self, data, rng):
        n_step = rng.randint(self.factor[0], self.factor[1])
        start_pos, end_pos = self.get_augment_range_by_coverage(data, rng)
        aug_data = librosa.effects.pitch_shift(data[start_pos:end_pos], sr=self.sampling_rate, n_steps=n_step)
        augmented_data = data.copy()
        augmented_data[start_pos:end_pos] = aug_data
        return augmented_data


def gen_augmented(original: np.ndarray, sample_rate: int, np_rng: np.random.RandomState, py_rng: random.Random) -> np.ndarray:
    """Original utils.py:184-197: pick one of the five uniformly and apply it once."""
    augment_list = [
        Noise(),
        Speed(),
        Loudness(factor=(0.5, 2)),
        Vtlp(sampling_rate=sample_rate, zone=(0.0, 1.0)),
        Pitch(sampling_rate=sample_rate, factor=(-1, 3)),
    ]
    aug_idx = py_rng.randint(0, len(augment_list) - 1)
    return augment_list[aug_idx].manipulate(np.asarray(original, dtype=np.float32), np_rng)
