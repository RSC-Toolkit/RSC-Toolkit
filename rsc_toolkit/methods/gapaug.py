"""
GaP-Aug(ICASSP 2024) Algorithm 1 의 배치 학습 절차 — **논문 명세 재구현** (코드 없음).

확정(논문 Algo. 1):
  7-8  : X̃_a = 홀수 위치 {x̃1, x̃3, …}, X̃_b = 짝수 위치 {x̃2, x̃4, …} (0-based 로는 a=짝수 인덱스, b=홀수 인덱스)
  9-12 : 쌍마다 λ 생성, X_mix = λX̃_a + (1-λ)X̃_b, Y_mix = λY_a + (1-λ)Y_b (soft label)
  13   : prediction = M(X_mix); embedding = M(X̃_a)
  14   : L = L_NLL(prediction, Y_mix) + L_triplet(embedding)

가정(저자 확인 전 — AUTHOR_QUESTIONS GaP-Aug #2, #3, #4):
- λ ~ Beta(α, α), α 미기재 → 1.0 — `mixup_alpha`. 근거: (i) mixup 원 논문 기본 범위, (ii) 백본 출처인 공식 PANNs 파이프라인이
  Mixup(mixup_alpha=1.)(outputs/reference/panns/pytorch/main.py:174) 을 쓰고 do_mixup(pytorch_utils.py:18-30) 이 짝/홀 인덱스 쌍을
  섞어 Algorithm 1 line 7-12 와 구조가 동일. 차이: PANNs 는 bn0 **뒤**(spec_augmenter 뒤) 에서 mixup 하지만 여기서는 bn0 앞의
  비음수화 스펙트로그램에서 섞는다(논문 Algo. 1 은 X̃ 스펙트로그램 단계) — 여전히 가정.
- soft-target NLL: -(Y_mix · log_softmax).sum(1).mean() — `mixup_loss="soft"`;
  대안 λ·NLL(y_a) + (1-λ)·NLL(y_b) — `mixup_loss="pair"` (단일 λ 가 아니라 쌍별 λ 로 가중).
- triplet: torch TripletMarginLoss(margin 1.0, p 2) 를 X̃_a 임베딩에 대해 배치 내 무작위 (anchor, positive, negative) 로
  구성 — anchor = X̃_a 의 각 샘플, positive = 같은 클래스의 다른 샘플(없으면 제외), negative = 다른 클래스 샘플 무작위.
  가중치: 논문 Algorithm 1 line 14 'L ← L_NLL(prediction, Y_mix) + L_triplet(embedding)' 은 **비가중 합** → 기본 w_trip = 1.0(문면).
  0.01 은 자매 논문(JMIR AI 2025, jmir_full.txt:533-534 L = L_CE + λ·L_triplet, λ=0.01; CE 기반·다른 세팅) 에서 가져온 **의도적 이탈
  플래그**(`w_trip=0.01`; 수정 회차 1 까지의 기본값). 두 값을 P1 에서 병기 판정한다.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def split_odd_even(n: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Algo. 1 line 7-8: a = {1,3,…,B-1}(1-based) → 0-based 짝수 인덱스, b = 홀수 인덱스. B 홀수면 마지막 하나는 버린다."""
    idx_a = torch.arange(0, n - (n % 2), 2, device=device)
    idx_b = idx_a + 1
    return idx_a, idx_b


def mixup_pairs(
    xa: torch.Tensor, xb: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor, n_cls: int, alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """쌍별 λ~Beta(α,α) (alpha<=0 이면 λ=1 → 혼합 없음). 반환 (x_mix, y_mix(soft, (n, n_cls)), lam(n,))."""
    n = xa.shape[0]
    if alpha > 0:
        # λ ~ Beta(α,α) — torch 전역 RNG 사용 (seed_everything 으로 고정)
        a = torch.tensor(float(alpha))
        lam = torch.distributions.Beta(a, a).sample((n,))
    else:
        lam = torch.ones(n)
    lam = lam.to(xa.device, xa.dtype)
    shape = (n,) + (1,) * (xa.dim() - 1)
    x_mix = lam.view(shape) * xa + (1.0 - lam.view(shape)) * xb
    oa = F.one_hot(ya, n_cls).to(xa.dtype)
    ob = F.one_hot(yb, n_cls).to(xa.dtype)
    y_mix = lam.view(n, 1) * oa + (1.0 - lam.view(n, 1)) * ob
    return x_mix, y_mix, lam



def soft_target_nll(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    """L_NLL(prediction, Y_mix): -(Y_mix · log_softmax(logits)).sum(1).mean()."""
    return -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def pair_nll(logits: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """대안: 쌍별 λ·NLL(y_a) + (1-λ)·NLL(y_b) (soft-target 과 수학적으로 동일하나 플래그로 둔다)."""
    lp = F.log_softmax(logits, dim=1)
    return (-(lam * lp.gather(1, ya[:, None]).squeeze(1)) - ((1 - lam) * lp.gather(1, yb[:, None]).squeeze(1))).mean()


def random_triplets(labels: torch.Tensor, gen: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    배치 내 무작위 (anchor, positive, negative) 인덱스 (가정). anchor = 모든 샘플; positive 는 같은 클래스의 다른 샘플에서
    무작위(없으면 그 anchor 제외); negative 는 다른 클래스에서 무작위(없으면 제외). 반환 각 (n_valid,) long (CPU 계산).
    """
    y = labels.detach().cpu()
    n = y.shape[0]
    A, P, N = [], [], []
    for i in range(n):
        pos = torch.nonzero((y == y[i]) & (torch.arange(n) != i)).flatten()
        neg = torch.nonzero(y != y[i]).flatten()
        if len(pos) == 0 or len(neg) == 0:
            continue
        j = pos[torch.randint(len(pos), (1,), generator=gen)].item()
        k = neg[torch.randint(len(neg), (1,), generator=gen)].item()
        A.append(i); P.append(j); N.append(k)
    dev = labels.device
    return torch.tensor(A, dtype=torch.long, device=dev), torch.tensor(P, dtype=torch.long, device=dev), torch.tensor(N, dtype=torch.long, device=dev)


def triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 1.0, p: float = 2.0,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """L_triplet(embedding): TripletMarginLoss(margin, p) 를 무작위 삼중항에 적용. 유효 삼중항이 없으면 0."""
    a, pidx, nidx = random_triplets(labels, gen)
    if a.numel() == 0:
        return emb.sum() * 0.0
    return F.triplet_margin_loss(emb[a], emb[pidx], emb[nidx], margin=margin, p=p)
