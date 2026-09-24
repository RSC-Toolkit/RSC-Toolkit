"""
Spectrogram augmentation of GaP-Aug (Chang et al., ICASSP 2024): **re-implemented from the paper specification** (no official or unofficial code).

Specified in paper §2.2.2 / Algorithm 1:
- For each sample (spectrogram), pick random patches P and apply p̃ = Gain·p^γ (gain 1) to each patch p.
- γ ~ U[1.7, 2.0], 32 patches, each patch at most 16 (frequency) × 16 (time).
- Table 2 baselines: SpecAug (time/frequency band masking), PatchMask (patches set to 0).

Assumptions (not confirmed by the authors):
- Non-negativity: log-mel values are negative, so the power is undefined. Default = apply γ to values min-max normalized to [0,1] per sample
  (normalization is done by the model's `norm_mode`; this module assumes non-negative input and applies clamp(min=0) **only to the patch region** before the power →
  even with norm_mode=none, values outside patches are unchanged and only negative values inside patches are lost to 0).
  Open question whether the model input after normalization is also on that scale: the default sample_minmax feeds [0,1] to the model;
  the alternative sample_minmax_restore applies only γ in [0,1] and restores the original dB scale via sample_minmax_split/sample_minmax_restore before the model.
- Patch height and width each uniform in 1..16, uniform position, overlaps allowed (applied sequentially → exponents multiply), resampled every iteration.
- γ independent per patch (gamma_per="patch"; "sample" gives one per sample), application probability p=1.0.
- SpecAug parameters not specified → since the backbone is PANNs, the PANNs SpecAugmentation defaults (time 64×2, freq 8×2) are assumed, mask value 0.
  Note on what this assumption means: the official Cnn14 applies SpecAugmentation **after** bn0, so 0 = BN-normalized mean; here it is applied to the [0,1]
  normalized input before bn0, so 0 = the per-sample minimum (quietest value) (to compare at the same position as GaP-Aug and PatchMask).
  time_drop_width of 64 frames is 0.64 s at the PANNs hop of 10 ms and 0.4 s at the paper preset (hop 6.25 ms); the frame count is kept.
  If the authors used Transfer_Cnn14 as-is, its built-in spec_augmenter may have been on for all Table 2 rows (including Naïve); unconfirmed.
- PatchMask = set patches to 0 using the same patch sampling rule as GaP-Aug (paper: 'randomly masks patches').

Input convention: (B, 1, T, F), the PANNs internal batch layout (time, mel). Height = frequency axis (F), width = time axis (T).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def sample_patches(
    n_patches: int, T: int, F: int, max_t: int, max_f: int, min_t: int, min_f: int, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    """(n_patches, 4) long: [t0, t1, f0, f1); uniform size (min..max, capped by the axis length), uniform position."""
    max_t, max_f = max(1, min(max_t, T)), max(1, min(max_f, F))
    min_t, min_f = max(1, min(min_t, max_t)), max(1, min(min_f, max_f))
    ht = torch.randint(min_t, max_t + 1, (n_patches,), generator=gen)
    hf = torch.randint(min_f, max_f + 1, (n_patches,), generator=gen)
    t0 = (torch.rand(n_patches, generator=gen) * (T - ht + 1).float()).long()
    f0 = (torch.rand(n_patches, generator=gen) * (F - hf + 1).float()).long()
    return torch.stack([t0, t0 + ht, f0, f0 + hf], dim=1)


class GammaPatchAug(nn.Module):
    """
    Applies p̃ = gain · p^γ to n_patches patches per sample (mode="gamma"), or sets the same patches to 0 (mode="mask" = PatchMask).
    Input (B,1,T,F) must be non-negative (log-mel normalized to [0,1] by norm_mode).
    """

    def __init__(
        self,
        n_patches: int = 32,
        max_freq: int = 16,
        max_time: int = 16,
        min_freq: int = 1,            # assumption: minimum 1
        min_time: int = 1,
        gamma_range: Tuple[float, float] = (1.7, 2.0),
        gain: float = 1.0,
        gamma_per: str = "patch",     # patch | sample (assumption)
        p: float = 1.0,               # per-sample application probability (assumed 1.0)
        mode: str = "gamma",          # gamma | mask
    ):
        super().__init__()
        if mode not in ("gamma", "mask"):
            raise ValueError(f"mode must be gamma|mask, got {mode!r}")
        if gamma_per not in ("patch", "sample"):
            raise ValueError(f"gamma_per must be patch|sample, got {gamma_per!r}")
        self.n_patches, self.max_freq, self.max_time = int(n_patches), int(max_freq), int(max_time)
        self.min_freq, self.min_time = int(min_freq), int(min_time)
        self.gamma_lo, self.gamma_hi = float(gamma_range[0]), float(gamma_range[1])
        self.gain, self.gamma_per, self.p, self.mode = float(gain), gamma_per, float(p), mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, gen: Optional[torch.Generator] = None, return_info: bool = False):
        if x.dim() != 4:
            raise ValueError(f"expected (B,1,T,F), got {tuple(x.shape)}")
        B, _, T, F = x.shape
        out = x.clone()
        info = []
        for b in range(B):
            if self.p < 1.0 and torch.rand((), generator=gen).item() >= self.p:
                info.append(None)
                continue
            patches = sample_patches(self.n_patches, T, F, self.max_time, self.max_freq, self.min_time, self.min_freq, gen)
            if self.gamma_per == "sample":
                g = torch.empty(1).uniform_(self.gamma_lo, self.gamma_hi, generator=gen).expand(self.n_patches).clone()
            else:
                g = torch.empty(self.n_patches).uniform_(self.gamma_lo, self.gamma_hi, generator=gen)
            for k in range(self.n_patches):
                t0, t1, f0, f1 = (int(v) for v in patches[k])
                if self.mode == "mask":
                    out[b, 0, t0:t1, f0:f1] = 0.0
                else:
                    region = out[b, 0, t0:t1, f0:f1].clamp(min=0.0)  # defensive: make only the patch region non-negative (norm_mode maps to [0,1])
                    out[b, 0, t0:t1, f0:f1] = self.gain * region.pow(float(g[k]))
            info.append({"patches": patches, "gamma": g})
        return (out, info) if return_info else out


class SpecAugPANNs(nn.Module):
    """
    Table 2 'SpecAug' baseline; parameters not specified, so the PANNs SpecAugmentation defaults are assumed
    (time_drop_width 64 ×2, freq_drop_width 8 ×2; each stripe width U[0, width), value 0). Implemented with torchaudio iid masking.
    Applied to the [0,1] input before bn0 (the official Cnn14 applies it after bn0); see the module docstring (assumption).
    Input (B,1,T,F) → internally permuted to (B,1,F,T) to match the torchaudio convention (…, freq, time).
    """

    def __init__(self, time_drop_width: int = 64, time_stripes: int = 2, freq_drop_width: int = 8, freq_stripes: int = 2, p: float = 1.0):
        super().__init__()
        from torchaudio import transforms as TA

        self.p = float(p)
        self.masks = nn.Sequential(
            *[TA.FrequencyMasking(freq_mask_param=freq_drop_width, iid_masks=True) for _ in range(freq_stripes)],
            *[TA.TimeMasking(time_mask_param=time_drop_width, iid_masks=True) for _ in range(time_stripes)],
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and torch.rand(()).item() >= self.p:
            return x
        return self.masks(x.transpose(2, 3)).transpose(2, 3).contiguous()


def normalize_nonneg(x: torch.Tensor, mode: str = "sample_minmax", db_offset: float = 100.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Makes log-mel non-negative (assumption). x: (B,1,T,F).
      sample_minmax : per-sample (x-min)/(max-min) → [0,1]   (default assumption)
      batch_minmax  : min-max over the whole batch → [0,1]
      db_offset     : (x + db_offset).clamp(min=0) / db_offset   (global dB offset; PANNs log-mel is roughly [-100, +40] dB)
      none          : unchanged (γ correction applies clamp(min=0) to the patch region only; negatives inside patches lost, outside unchanged; for comparison)
      sample_minmax_restore : evaluation input stays in the original dB (= none). During training only the augmentation runs in [0,1], then restored; see make_augmented_input in the model
    """
    if mode == "sample_minmax_restore":
        return x
    if mode == "sample_minmax":
        mn = x.amin(dim=(1, 2, 3), keepdim=True)
        mx = x.amax(dim=(1, 2, 3), keepdim=True)
        return (x - mn) / (mx - mn + eps)
    if mode == "batch_minmax":
        return (x - x.min()) / (x.max() - x.min() + eps)
    if mode == "db_offset":
        return (x + db_offset).clamp(min=0.0) / db_offset
    if mode == "none":
        return x
    raise ValueError(f"unknown norm mode {mode!r}")


def sample_minmax_split(x: torch.Tensor, eps: float = 1e-8):
    """Per-sample min-max: (x01, mn, mx), for sample_minmax_restore. x01 = (x-mn)/(mx-mn+eps)."""
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - mn) / (mx - mn + eps), mn, mx


def sample_minmax_restore(x01: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[0,1] → restores the original dB scale: x01·(mx-mn+eps) + mn (patches reduced by γ move toward mn)."""
    return x01 * (mx - mn + eps) + mn
