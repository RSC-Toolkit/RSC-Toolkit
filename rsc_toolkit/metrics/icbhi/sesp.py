"""Se / Sp / Score 산식의 단일 출처 (리팩토링 P2-5).

ICBHI 관례: S_p = normal 클래스 정확도, S_e = 비정상 클래스 전체 정확도(클래스별 hit 합 / count 합), Score = 평균.
같은 식이 ICBHIScore.compute · get_score(BTS 이식) · MVST 융합 · RespireNet 투표 · eval 스크립트에 따로 있었고
분모 보호(eps vs max(·,1))·클래스 수 결정이 구현마다 달랐다. 툴킷 경로(ICBHIScore·get_score)는 이 함수를 쓴다.

원본 코드를 축자 이식한 경로는 **골든 패리티** 때문에 그대로 둔다 — methods/mvst_fusion.get_score(분모 보호 없음),
trainer/callbacks/respirenet_vote_eval.vote_cycles(max(·,1), 관측값 기반 n_cls), scripts/reproduction/eval_*_checkpoint.py.
그 값들은 count 가 0 이 아닌 한 이 함수와 1e-10 상대 오차 안에서 같다.
"""
from __future__ import annotations

from typing import Sequence, Tuple

EPS = 1e-10  # BTS util/icbhi_util.py get_score 의 분모 보호 — ICBHIScore 도 같은 값을 써 왔다


def sesp_score(hits: Sequence[float], counts: Sequence[float], *, normal_index: int = 0, eps: float = EPS) -> Tuple[float, float, float]:
    """클래스별 (hits, counts) → (S_p, S_e, Score) 백분율. `normal_index` 가 S_p 의 클래스, 나머지 전부가 S_e."""
    if len(hits) != len(counts):
        raise ValueError(f"hits 와 counts 의 길이가 다르다: {len(hits)} vs {len(counts)}")
    if not 0 <= normal_index < len(counts):
        raise ValueError(f"normal_index={normal_index} 가 클래스 수 {len(counts)} 를 벗어난다")
    sp = float(hits[normal_index]) / (float(counts[normal_index]) + eps) * 100.0
    ab_hits = sum(float(h) for i, h in enumerate(hits) if i != normal_index)
    ab_counts = sum(float(c) for i, c in enumerate(counts) if i != normal_index)
    se = ab_hits / (ab_counts + eps) * 100.0
    return sp, se, (sp + se) / 2.0


__all__ = ["sesp_score", "EPS"]
