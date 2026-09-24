"""
Lungmix (ICASSP 2025) loss: **re-implemented from the paper specification** (no loss code in the official repository; paper §II-C Eqs. 9–10).

variant:
- non_linear        : L = L_CE(g(x̃), ỹ), ỹ = y_i ⊕ y_j (OR label). Table I 'Lungmix (non-linear)'; the main method.
- linear            : L = L_Mixup = λ·CE(g(x̃), y_1) + (1−λ)·CE(g(x̃), y_2) (Eq. 9). Table I 'Lungmix (linear)'.
- combined          : L = λ1·L_CE(ỹ) + λ2·L_Mixup, λ1 = 1 (Eq. 10). λ2 = 'rescaled according to the first loss' is undefined →
                      **Assumption**: λ2 = L_CE.detach() / L_Mixup.detach() (a rescale that matches the magnitudes of the two terms).
                      A constant is also possible with lambda2_mode='fixed' + lambda2.
- label_preservation: L = CE(g(x̃), y_1) (keeps the first input's label for the Lungmix waveform input, as defined in §II-C). The 'Patch-mix (label
                      preservation)' row of Table I preserves labels for Patch-mix input, so **no Table I row corresponds to this variant** (exploratory).
- vanilla_mixup     : loss identical to linear (Eq. 9); input mixing is linear waveform mixup (in the model).
- none              : L = CE(logits, y); the 'w/o Mixup' baseline.
- lungmix_patchmix  : loss is PatchMixLoss (OR label) inside the model; not handled like non_linear here, the model computes it directly.

λ differs per pair (the official lungmix() draws Beta(0.5,0.5) on every call) → Eq. 9 is weighted by per-sample λ (reduction='none', then mean).
If the criterion has a class weight it is used as-is (the paper does not mention weighted CE → unweighted CE by default).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("none", "non_linear", "linear", "combined", "label_preservation", "vanilla_mixup", "lungmix_patchmix")


def _per_sample_ce(logits: torch.Tensor, y: torch.Tensor, criterion: Optional[nn.Module]) -> torch.Tensor:
    weight = getattr(criterion, "weight", None) if criterion is not None else None
    ls = float(getattr(criterion, "label_smoothing", 0.0) or 0.0) if criterion is not None else 0.0
    return F.cross_entropy(logits, y, weight=weight, reduction="none", label_smoothing=ls)


def mixup_ce(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: torch.Tensor,
             criterion: Optional[nn.Module] = None) -> torch.Tensor:
    """Eq. 9 with per-sample λ. lam: (B,) or scalar."""
    lam = torch.as_tensor(lam, dtype=logits.dtype, device=logits.device).reshape(-1)
    if lam.numel() == 1:
        lam = lam.expand(logits.shape[0])
    return (lam * _per_sample_ce(logits, y_a, criterion) + (1.0 - lam) * _per_sample_ce(logits, y_b, criterion)).mean()


class LungmixLoss(nn.Module):
    def __init__(self, variant: str = "non_linear", lambda1: float = 1.0, lambda2_mode: str = "ratio",
                 lambda2: float = 1.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}: {variant!r}")
        if lambda2_mode not in ("ratio", "fixed"):
            raise ValueError("lambda2_mode must be ratio|fixed")
        self.variant, self.lambda1, self.lambda2_mode, self.lambda2 = variant, float(lambda1), lambda2_mode, float(lambda2)

    def forward(self, logits, y_or, y_a, y_b, lam, criterion: Optional[nn.Module] = None):
        ce = criterion if criterion is not None else nn.CrossEntropyLoss()
        v = self.variant
        if v in ("none",):
            return ce(logits, y_a)
        if v in ("non_linear", "lungmix_patchmix"):
            return ce(logits, y_or)
        if v in ("linear", "vanilla_mixup"):
            return mixup_ce(logits, y_a, y_b, lam, criterion)
        if v == "label_preservation":
            return ce(logits, y_a)
        # combined (Eq. 10)
        l_ce = ce(logits, y_or)
        l_mix = mixup_ce(logits, y_a, y_b, lam, criterion)
        if self.lambda2_mode == "ratio":
            lam2 = (l_ce.detach() / l_mix.detach().clamp_min(1e-8))
        else:
            lam2 = torch.as_tensor(self.lambda2, dtype=l_ce.dtype, device=l_ce.device)
        return self.lambda1 * l_ce + lam2 * l_mix
