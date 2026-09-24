"""Single source of the Se / Sp / Score formula.

ICBHI convention: S_p = accuracy on the normal class, S_e = accuracy over all abnormal classes (sum of per-class hits / sum of counts), Score = mean.
The toolkit paths (ICBHIScore, get_score) use this function.

Paths ported verbatim from the official code keep their own formula for **golden parity** — methods/mvst_fusion.get_score (no denominator guard),
trainer/callbacks/respirenet_vote_eval.vote_cycles (max(·,1), n_cls from observed labels), scripts/reproduction/eval_*_checkpoint.py.
Their values agree with this function within 1e-10 relative error as long as no count is 0.
"""
from __future__ import annotations

from typing import Sequence, Tuple

EPS = 1e-10  # denominator guard of get_score in BTS util/icbhi_util.py — ICBHIScore uses the same value


def sesp_score(hits: Sequence[float], counts: Sequence[float], *, normal_index: int = 0, eps: float = EPS) -> Tuple[float, float, float]:
    """Per-class (hits, counts) → (S_p, S_e, Score) in percent. `normal_index` is the S_p class; all others form S_e."""
    if len(hits) != len(counts):
        raise ValueError(f"hits and counts differ in length: {len(hits)} vs {len(counts)}")
    if not 0 <= normal_index < len(counts):
        raise ValueError(f"normal_index={normal_index} is out of range for {len(counts)} classes")
    sp = float(hits[normal_index]) / (float(counts[normal_index]) + eps) * 100.0
    ab_hits = sum(float(h) for i, h in enumerate(hits) if i != normal_index)
    ab_counts = sum(float(c) for i, c in enumerate(counts) if i != normal_index)
    se = ab_hits / (ab_counts + eps) * 100.0
    return sp, se, (sp + se) / 2.0


__all__ = ["sesp_score", "EPS"]
