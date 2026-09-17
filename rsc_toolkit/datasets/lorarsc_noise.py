"""
**Alternative interpretation** of the LoRA-RSC (ICASSP 2025) waveform noise (1D waveform -> per-sample N(0, 0.3^2) Gaussian noise) -- IndexDataset subclass behind a flag.

The public code dataset/icbhi.py:241-245 adds `np.random.randn(len(audio_numpy))` to a (1,N) 2D array, so len=1 -> a **scalar DC offset** per clip,
which kaldi fbank's remove_dc_offset removes, leaving the features unchanged (golden features.pt, max|diff| 6.4e-5). However, as orig_2d_call in the same
golden shows, the public code **cannot run at all** with this (1,N) call (4.28.1 ASTFeatureExtractor -> kaldi AssertionError), so
the authors' actual run must have differed from the public code at this point. The most natural fix (making it 1D, e.g. audio.numpy()[0]) gives len=128000 -> per-sample
N(0,0.3^2) noise baked once (fixed) into the train/test cache, an entirely different pipeline. The code cannot settle which one was used -> the default config (icbhi_lorarsc) has no noise (scalar interpretation = no-op), and this class + data=icbhi_lorarsc_noise
is the 1D interpretation.

Reproduction approach: the original draws the noise once in dataset order under np.random.seed(10) and caches it (depends on global RNG order, not exactly reproducible), so
here a per-sample RNG determined by sample_id and noise_seed reproduces only the **cached once, shared by train/test, deterministic** properties.
Adding in float64 and then casting to float32 follows the original order (astype(float64) -> float32 inside the extractor).
The feature cache key is built from the processor config only, so the noisy version must use a **different cache_dir** (see icbhi_lorarsc_noise.yaml).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np
import torch

from .index_dataset import IndexDataset


class LoRARSCNoisyIndexDataset(IndexDataset):
    """IndexDataset + per-sample Gaussian waveform noise (std=noise_std, deterministic).

    Controlled by the constructor arguments `noise_std`/`noise_seed` -- in a config: `data.params.index_dataset_params: {noise_std: 0.3, noise_seed: 10}`.
    The class attributes are the defaults.
    """

    noise_std: float = 0.3      # dataset/icbhi.py:243 `0.3 * noise`
    noise_seed: int = 10        # original --seed default 10 (np.random.seed(10)); base of the per-sample derived seeds

    def __init__(self, *args, noise_std: float | None = None, noise_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if noise_std is not None:
            self.noise_std = float(noise_std)
        if noise_seed is not None:
            self.noise_seed = int(noise_seed)

    @classmethod
    def sample_rng(cls, sample_id: str, noise_seed: int | None = None) -> np.random.Generator:
        seed = cls.noise_seed if noise_seed is None else int(noise_seed)
        h = hashlib.md5(f"{seed}:{sample_id}".encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    @classmethod
    def add_noise(cls, audio: torch.Tensor, sample_id: str, noise_std: float | None = None,
                  noise_seed: int | None = None) -> torch.Tensor:
        std = cls.noise_std if noise_std is None else float(noise_std)
        if std <= 0:
            return audio
        x = audio.detach().cpu().numpy().astype(np.float64)
        noise = cls.sample_rng(sample_id, noise_seed).standard_normal(size=x.shape)  # 1D interpretation: independent N(0,1) per sample
        return torch.from_numpy((x + std * noise).astype(np.float32))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        out["audio"] = self.add_noise(out["audio"], out["sample_id"], self.noise_std, self.noise_seed)
        return out
