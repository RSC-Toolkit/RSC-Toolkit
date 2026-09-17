"""
**Binary (non-standard)** Se/Sp/Score of the AMS-D paper/official code — port of calculate_stats in utils/metric.py:18-34.

  Se = #(y≠0 ∧ ŷ≠0) / #abnormal    (an abnormal sample counts as correct if predicted as any abnormal class)
  Sp = #(y=0 ∧ ŷ=0) / #normal
  Score = (Se + Sp) / 2            (main.py:416)
  acc = 4-class accuracy           (for reference)

This is the formula of paper §3.1, "Sen./Spe. are computed in a binary setting by grouping all abnormal classes as positive".
It is **not comparable** with the standard ICBHI 4-class Score (rsc_toolkit/metrics/icbhi/score.py, only the exact abnormal class counts
as correct) — the keys are named "paper_S_e"/"paper_S_p"/"paper_score" and tables label it "paper metric (non-standard)". Values are in %.
"""
from typing import Dict

import torch
from torchmetrics import Metric


class PaperBinaryScore(Metric):
    full_state_update = False

    def __init__(self, normal_index: int = 0, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.normal_index = int(normal_index)
        self.add_state("tp", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("num_pos", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("num_neg", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, C], got {logits.shape}")
        target = target.view(-1)
        pred = logits.argmax(dim=1)
        is_pos = target != self.normal_index
        self.num_pos += is_pos.sum()
        self.num_neg += (~is_pos).sum()
        self.tp += (is_pos & (pred != self.normal_index)).sum()
        self.tn += ((~is_pos) & (pred == self.normal_index)).sum()
        self.correct += (pred == target).sum()
        self.total += target.numel()

    def compute(self) -> Dict[str, torch.Tensor]:
        # original code: 0 if num_pos/num_neg is 0 (utils/metric.py:30-31)
        se = self.tp.float() / self.num_pos.float() * 100.0 if int(self.num_pos) else self.tp.float() * 0.0
        sp = self.tn.float() / self.num_neg.float() * 100.0 if int(self.num_neg) else self.tn.float() * 0.0
        acc = self.correct.float() / self.total.float().clamp_min(1) * 100.0
        return {"paper_S_p": sp, "paper_S_e": se, "paper_score": (se + sp) / 2.0, "paper_acc": acc}
