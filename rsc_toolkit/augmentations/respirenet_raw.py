"""
RespireNet 학습 파형 증강 — 원본 utils.py gen_augmented 가 쓰는 nlpaug 0.0.14 오디오 증강 5종의 충실 이식.

출처: microsoft/RespireNet utils.py:184-197 + nlpaug 0.0.14 nlpaug/model/audio/{audio,noise,speed,loudness,vtlp,pitch}.py.
원본 호출: NoiseAug(), SpeedAug(), LoudnessAug(factor=(0.5,2)), VtlpAug(sr, zone=(0,1)), PitchAug(sr, factor=(-1,3))
중 하나를 random.randint 로 고르고 augment() (= model.manipulate) 1회.

nlpaug 0.0.14 의 실제 동작을 그대로 옮긴다 (의도가 아니라 동작이 논문 수치 조건이므로):
- zone 기본 (0.2, 0.8)·coverage 1 → 가운데 60% 구간만 증강 (Vtlp 만 zone (0,1)·coverage 0.1).
- Speed: {0.5..1.9}\\{1.0} 0.1 간격, Pitch: randint(-1,3) → {-1,0,1,2} 반음, Loudness: uniform(0.5,2).
- Noise: 백색잡음 irfft(randn) 을 그대로 더함(스케일 없음), 구간 위치는 randint 로 패딩.
- Vtlp: librosa.stft 출력(freq, frames)을 (time, freq) 로 오독해 **프레임 축을 워핑**하는 nlpaug 버그 포함.
- 원본 nlpaug 는 librosa ≥0.10 에서 time_stretch/pitch_shift 위치 인자로 TypeError — 저자 환경(librosa 0.8)에서는
  동작했으므로 키워드 인자로 같은 연산을 호출한다.
난수: np.random → 전달받은 RandomState, random → 전달받은 random.Random (재현성).
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
        # color == 'white': 색 가중 없음
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
        time_dim, freq_dim = stft.shape  # nlpaug 의 축 오독 그대로 (실제로는 freq, frames)
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
    """원본 utils.py:184-197 — 5종 중 하나를 균등 선택해 1회 적용."""
    augment_list = [
        Noise(),
        Speed(),
        Loudness(factor=(0.5, 2)),
        Vtlp(sampling_rate=sample_rate, zone=(0.0, 1.0)),
        Pitch(sampling_rate=sample_rate, factor=(-1, 3)),
    ]
    aug_idx = py_rng.randint(0, len(augment_list) - 1)
    return augment_list[aug_idx].manipulate(np.asarray(original, dtype=np.float32), np_rng)
