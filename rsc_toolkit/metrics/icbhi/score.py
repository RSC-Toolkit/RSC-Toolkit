# Portions of this file are ported from BTS (https://github.com/kaen2891/bts), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Dict

import torch
from torchmetrics import Metric

from rsc_toolkit.metrics.icbhi.sesp import EPS, sesp_score


def get_score(hits, counts, pflag=False):
    """Same values as get_score in BTS util/icbhi_util.py (normal = class 0). The formula lives in sesp_score only."""
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

    Output keys default to {"S_p", "S_e", "icbhi_score"}. For other datasets such as SPRSound the score can be renamed with
    `score_key="sprsound_score"`, and `key_prefix` avoids collisions when two instances are used side by side (the best-checkpoint
    callback's score_key, the best.json schema, and the verification scripts assume the default keys).
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
        """Key names produced by compute() — callbacks read them here instead of guessing the strings."""
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
                # Labels outside [0, num_classes) would be silently dropped by the per-class loop: feeding 7-class labels to a
                # 4-class metric would compute the Score from the remaining labels only and yield values close to 100.
                raise ValueError(
                    f"ICBHIScore(num_classes={self.num_classes}) received out-of-range labels: min {lo}, max {hi}. "
                    f"Check that num_classes in the metrics config matches the dataset label space (e.g. SPRSound 7-class vs 7→4 merged)."
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
        # Tensor arithmetic with the same formula as sesp_score (eps 1e-10), using the torch.float32 state as is —
        # the result must stay bit-identical so that best.json and golden values remain valid.
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
