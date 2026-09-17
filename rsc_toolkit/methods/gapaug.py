"""
Batch training procedure of GaP-Aug (ICASSP 2024) Algorithm 1: **re-implemented from the paper specification** (no code released).

Specified (paper Algo. 1):
  7-8  : X̃_a = odd positions {x̃1, x̃3, …}, X̃_b = even positions {x̃2, x̃4, …} (0-based: a = even indices, b = odd indices)
  9-12 : λ drawn per pair, X_mix = λX̃_a + (1-λ)X̃_b, Y_mix = λY_a + (1-λ)Y_b (soft label)
  13   : prediction = M(X_mix); embedding = M(X̃_a)
  14   : L = L_NLL(prediction, Y_mix) + L_triplet(embedding)

Assumptions (not confirmed by the authors):
- λ ~ Beta(α, α), α not specified → 1.0 (`mixup_alpha`). Rationale: (i) default range of the original mixup paper, (ii) the official PANNs pipeline the backbone comes from
  uses Mixup(mixup_alpha=1.) (panns pytorch/main.py:174), and do_mixup (pytorch_utils.py:18-30) mixes even/odd index pairs,
  structurally identical to Algorithm 1 lines 7-12. Difference: PANNs mixes **after** bn0 (after spec_augmenter), whereas here the mix happens on the
  non-negative spectrogram before bn0 (paper Algo. 1 mixes at the X̃ spectrogram stage); still an assumption.
- soft-target NLL: -(Y_mix · log_softmax).sum(1).mean() — `mixup_loss="soft"`;
  alternative λ·NLL(y_a) + (1-λ)·NLL(y_b) — `mixup_loss="pair"` (weighted by per-pair λ, not a single λ).
- triplet: torch TripletMarginLoss(margin 1.0, p 2) on the X̃_a embeddings with random in-batch (anchor, positive, negative)
  triplets: anchor = each sample of X̃_a, positive = another sample of the same class (excluded if none), negative = random sample of another class.
  Weight: paper Algorithm 1 line 14 'L ← L_NLL(prediction, Y_mix) + L_triplet(embedding)' is an **unweighted sum** → default w_trip = 1.0 (literal).
  0.01 comes from a sister paper (JMIR AI 2025, L = L_CE + λ·L_triplet, λ=0.01; CE-based, different setting) and is an **intentional deviation
  flag** (`w_trip=0.01`). Both values are reported in the 5-seed reproduction check.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def split_odd_even(n: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Algo. 1 line 7-8: a = {1,3,…,B-1} (1-based) → 0-based even indices, b = odd indices. If B is odd, the last sample is dropped."""
    idx_a = torch.arange(0, n - (n % 2), 2, device=device)
    idx_b = idx_a + 1
    return idx_a, idx_b


def mixup_pairs(
    xa: torch.Tensor, xb: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor, n_cls: int, alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pair λ~Beta(α,α) (alpha<=0 gives λ=1 → no mixing). Returns (x_mix, y_mix(soft, (n, n_cls)), lam(n,))."""
    n = xa.shape[0]
    if alpha > 0:
        # λ ~ Beta(α,α) using the torch global RNG (fixed by seed_everything)
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
    """Alternative: per-pair λ·NLL(y_a) + (1-λ)·NLL(y_b) (mathematically identical to soft-target, kept as a flag)."""
    lp = F.log_softmax(logits, dim=1)
    return (-(lam * lp.gather(1, ya[:, None]).squeeze(1)) - ((1 - lam) * lp.gather(1, yb[:, None]).squeeze(1))).mean()


def random_triplets(labels: torch.Tensor, gen: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random in-batch (anchor, positive, negative) indices (assumption). anchor = every sample; positive is random among other samples
    of the same class (anchor excluded if none); negative is random from other classes (excluded if none). Returns (n_valid,) long each (computed on CPU).
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
    """L_triplet(embedding): TripletMarginLoss(margin, p) on random triplets. 0 if there are no valid triplets."""
    a, pidx, nidx = random_triplets(labels, gen)
    if a.numel() == 0:
        return emb.sum() * 0.0
    return F.triplet_margin_loss(emb[a], emb[pidx], emb[nidx], margin=margin, p=p)
