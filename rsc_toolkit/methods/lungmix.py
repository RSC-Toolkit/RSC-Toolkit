"""
Lungmix(ICASSP 2025) 손실 — **논문 명세 재구현** (공식 저장소에 손실 코드 없음; 논문 §II-C 식 9–10).

variant:
- non_linear        : L = L_CE(g(x̃), ỹ), ỹ = y_i ⊕ y_j (OR 라벨). Table I 'Lungmix (non-linear)' — 주 방법.
- linear            : L = L_Mixup = λ·CE(g(x̃), y_1) + (1−λ)·CE(g(x̃), y_2) (식 9). Table I 'Lungmix (linear)'.
- combined          : L = λ1·L_CE(ỹ) + λ2·L_Mixup, λ1 = 1 (식 10). λ2 = 'rescaled according to the first loss' 는 미정의 →
                      **가정(AUTHOR_QUESTIONS #3)**: λ2 = L_CE.detach() / L_Mixup.detach() (두 항의 크기를 맞추는 rescale).
                      lambda2_mode='fixed' + lambda2 로 상수도 가능.
- label_preservation: L = CE(g(x̃), y_1) (Lungmix 파형 입력에 첫 입력 라벨 유지, §II-C 정의). Table I 의 'Patch-mix (label
                      preservation)' 행은 Patch-mix 입력에 라벨 보존이라 **이 variant 에 대응하는 Table I 행은 없다**(탐색용).
- vanilla_mixup     : 손실은 linear 와 동일(식 9); 입력 믹싱이 파형 선형 mixup (모델 쪽).
- none              : L = CE(logits, y) — 'w/o Mixup' 기준선.
- lungmix_patchmix  : 손실은 모델 쪽에서 PatchMixLoss(OR 라벨) — 여기서는 non_linear 와 같이 처리되지 않고 모델이 직접 계산.

λ 는 쌍마다 다르다(공식 lungmix() 가 호출마다 Beta(0.5,0.5) 를 뽑음) → 식 9 를 샘플별 λ 로 가중(reduction='none' 후 평균).
criterion 의 class weight 가 있으면 그대로 쓴다(논문은 가중 CE 언급 없음 → 기본 비가중 CE).
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
    """식 9, 샘플별 λ. lam: (B,) 또는 스칼라."""
    lam = torch.as_tensor(lam, dtype=logits.dtype, device=logits.device).reshape(-1)
    if lam.numel() == 1:
        lam = lam.expand(logits.shape[0])
    return (lam * _per_sample_ce(logits, y_a, criterion) + (1.0 - lam) * _per_sample_ce(logits, y_b, criterion)).mean()


class LungmixLoss(nn.Module):
    def __init__(self, variant: str = "non_linear", lambda1: float = 1.0, lambda2_mode: str = "ratio",
                 lambda2: float = 1.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant 는 {VARIANTS} 중 하나여야 한다: {variant!r}")
        if lambda2_mode not in ("ratio", "fixed"):
            raise ValueError("lambda2_mode 는 ratio|fixed")
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
        # combined (식 10)
        l_ce = ce(logits, y_or)
        l_mix = mixup_ce(logits, y_a, y_b, lam, criterion)
        if self.lambda2_mode == "ratio":
            lam2 = (l_ce.detach() / l_mix.detach().clamp_min(1e-8))
        else:
            lam2 = torch.as_tensor(self.lambda2, dtype=l_ce.dtype, device=l_ce.device)
        return self.lambda1 * l_ce + lam2 * l_mix
