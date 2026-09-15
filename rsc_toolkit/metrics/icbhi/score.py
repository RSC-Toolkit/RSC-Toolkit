from typing import Dict

import torch
from torchmetrics import Metric

from rsc_toolkit.metrics.icbhi.sesp import EPS, sesp_score


def get_score(hits, counts, pflag=False):
    """BTS util/icbhi_util.py get_score 와 같은 값 (normal = 클래스 0). 산식은 sesp_score 한 곳에 있다."""
    sp, se, sc = sesp_score(hits, counts, normal_index=0, eps=EPS)
    if pflag:
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))
    return sp, se, sc


class ICBHIScore(Metric):
    """
    ICBHI score implementation adapted and modified from BTS (get_score function).
    https://github.com/RSC-Toolkit/BTS/blob/8ec14c7145795858d414b5430e8ae8c218f0e284/util/icbhi_util.py#L520

    computes:
      S_p = acc on normal class (normal_index)
      S_e = acc on all other classes
      Score = (S_p + S_e)/2

    출력 키는 기본 {"S_p", "S_e", "icbhi_score"} 다. SPRSound 처럼 다른 데이터셋에 쓸 때 `score_key="sprsound_score"`
    로 이름을 바꾸거나, 같은 클래스 두 개를 나란히 둘 때 `key_prefix` 로 충돌을 피할 수 있다 (기본값이면 예전과 같다 —
    best-checkpoint 콜백의 score_key·best.json 스키마·검증 스크립트가 기본 키를 전제한다).
    """
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        normal_index: int = 0,
        topk: int = 1,
        dist_sync_on_step: bool = False,
        score_key: str = "icbhi_score",
        key_prefix: str = "",
        ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.normal_index = normal_index
        self.topk = topk
        self.score_key = str(score_key)
        self.key_prefix = str(key_prefix)

        # Per-class totals and hits (correct)
        self.add_state("counts", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("hits",   default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    @property
    def keys(self) -> Dict[str, str]:
        """compute() 가 내는 키 이름 — 콜백이 문자열을 알아맞히는 대신 여기서 읽는다."""
        p = self.key_prefix
        return {"S_p": f"{p}S_p", "S_e": f"{p}S_e", "score": f"{p}{self.score_key}"}

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """
        logits: [B, C]
        target: [B]
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, C], got {logits.shape}")
        if target.ndim != 1:
            target = target.view(-1)

        B, C = logits.shape
        k = min(self.topk, C)

        if target.numel() > 0:
            lo, hi = int(target.min()), int(target.max())
            if lo < 0 or hi >= self.num_classes:
                # 예전엔 [0, num_classes) 밖 라벨을 per-class 루프가 조용히 버려, 7-class 라벨을 4-class 메트릭에 넣으면
                # 남은 라벨만으로 Score 가 계산돼 100 에 가까운 값이 나왔다 (REVIEW 2026-09-10 §4 P2).
                raise ValueError(
                    f"ICBHIScore(num_classes={self.num_classes}) 에 범위 밖 라벨이 들어왔다: min {lo}, max {hi}. "
                    f"metrics config 의 num_classes 와 데이터셋 라벨 공간(예: SPRSound 7-class vs 7→4 병합)이 맞는지 확인하라."
                )

        # top-k predicted indices: [B, k]
        pred = logits.topk(k, dim=1, largest=True, sorted=True).indices

        # correct if target appears in top-k
        correct = (pred == target.unsqueeze(1)).any(dim=1)  # [B] bool

        # update per-class counts/hits
        for cls in range(self.num_classes):
            mask = (target == cls)
            self.counts[cls] += mask.sum()
            self.hits[cls] += (correct & mask).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        # 텐서 산술로 계산하되 산식은 sesp_score 와 동일(eps 1e-10). torch.float32 상태를 그대로 쓴다 —
        # 예전 compute 와 비트 단위로 같아야 best.json·골든 값이 유지된다.
        counts = self.counts.float()
        hits = self.hits.float()

        normal = self.normal_index
        eps = EPS

        sp = hits[normal] / (counts[normal] + eps) * 100.0

        abnormal_mask = torch.ones(self.num_classes, dtype=torch.bool, device=counts.device)
        abnormal_mask[normal] = False

        se = hits[abnormal_mask].sum() / (counts[abnormal_mask].sum() + eps) * 100.0
        score = (sp + se) / 2.0

        k = self.keys
        return {k["S_p"]: sp, k["S_e"]: se, k["score"]: score}
