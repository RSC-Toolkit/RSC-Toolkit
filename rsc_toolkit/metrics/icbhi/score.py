from typing import Dict

import torch
from torchmetrics import Metric


def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
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
    """
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        normal_index: int = 0,
        topk: int = 1,
        dist_sync_on_step: bool = False,
        ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.normal_index = normal_index
        self.topk = topk

        # Per-class totals and hits (correct)
        self.add_state("counts", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("hits",   default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

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
        counts = self.counts.float()
        hits = self.hits.float()

        normal = self.normal_index
        eps = 1e-10

        sp = hits[normal] / (counts[normal] + eps) * 100.0

        abnormal_mask = torch.ones(self.num_classes, dtype=torch.bool, device=counts.device)
        abnormal_mask[normal] = False

        se = hits[abnormal_mask].sum() / (counts[abnormal_mask].sum() + eps) * 100.0
        score = (sp + se) / 2.0

        return {"S_p": sp, "S_e": se, "icbhi_score": score}
