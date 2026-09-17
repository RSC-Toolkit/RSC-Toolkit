# Portions of this file are ported from BTS (https://github.com/kaen2891/bts), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Dict, Tuple

import torch
from torchmetrics import Metric


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        n_cls = output.shape[1]
        valid_topk = [k for k in topk if k <= n_cls]
        
        maxk = max(valid_topk)
        bsz = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k in valid_topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / bsz))
            else: res.append(torch.tensor([0.]))

        return res, bsz


class TopKAccuracy(Metric):
    """
    Stateful Top-K accuracy metric.

    ICBHI score implementation adapted and modified from BTS (accuracy function).
    https://github.com/RSC-Toolkit/BTS/blob/8ec14c7145795858d414b5430e8ae8c218f0e284/util/misc.py#L121

    - Accumulates correct predictions and total samples across batches.
    - Supports multiple K values (e.g. topk=(1, 3, 5)).
    - Returns percentages in [0, 100].
    """

    full_state_update = False

    def __init__(
        self,
        topk: Tuple[int, ...] = (1,),
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if len(topk) == 0:
            raise ValueError("topk must contain at least one value")
        if any(k <= 0 for k in topk):
            raise ValueError(f"all topk values must be > 0, got {topk}")

        # Keep as tuple, sorted & unique for consistent outputs
        self.topk = tuple(sorted(set(int(k) for k in topk)))

        # States:
        # correct[k] = number of samples correct within top-k (summed over batches)
        # total = number of samples
        self.add_state(
            "correct",
            default=torch.zeros(len(self.topk), dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """
        logits: [B, C]
        target: [B] (class indices)
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, C], got {logits.shape}")
        if target.ndim != 1:
            target = target.view(-1)

        B, C = logits.shape
        if B == 0:
            return

        maxk = min(max(self.topk), C)

        # pred: [B, maxk]
        pred = logits.topk(maxk, dim=1, largest=True, sorted=True).indices

        # correct_matrix: [B, maxk] whether each rank matches target
        correct_matrix = pred.eq(target.unsqueeze(1))

        # For each k: count samples where any of top-k matches target
        for i, k in enumerate(self.topk):
            kk = min(k, C)  # if k > num_classes, clamp
            correct_any = correct_matrix[:, :kk].any(dim=1)
            self.correct[i] += correct_any.sum()

        self.total += B

    def compute(self) -> Dict[str, torch.Tensor]:
        total = self.total.float().clamp_min(1.0)
        accs = self.correct.float() / total * 100.0

        return {f"acc@{k}": accs[i] for i, k in enumerate(self.topk)}
