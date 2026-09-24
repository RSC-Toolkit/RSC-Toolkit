"""
Training loss implementations (registered in loss_registry).
"""
from typing import Sequence

import torch
import torch.nn as nn


class WeightedCrossEntropyFromCounts(nn.Module):
    """
    CE with weights derived from per-class sample counts: the patch-mix family --weighted_loss.

    Source: raymin0223/patch-mix main.py:264-269
      weights = 1 / (counts / counts.sum());  weights /= weights.sum()

    class_nums example (ICBHI official train, 4-class): [2063, 1215, 501, 363]
    (re-checked against the .datasets/icbhi index + official split; total 4,142).
    """

    def __init__(self, class_nums: Sequence[float], **ce_kwargs):
        super().__init__()
        counts = torch.tensor(list(class_nums), dtype=torch.float32)
        weights = 1.0 / (counts / counts.sum())
        weights = weights / weights.sum()
        self.register_buffer("weight", weights)
        self.ce = nn.CrossEntropyLoss(weight=weights, **ce_kwargs)

    def forward(self, pred, target):
        return self.ce(pred, target)


class WeightedCrossEntropyFromPrior(nn.Module):
    """
    Weighted CE from RespireNet train.py:140-146:
      w = prior / prior.sum();  w = 1 / w;  w = w / w.sum()
    prior [3, 1, 1, 1] → [0.1, 0.3, 0.3, 0.3].
    """

    def __init__(self, prior: Sequence[float], **ce_kwargs):
        super().__init__()
        w = torch.tensor(list(prior), dtype=torch.float32)
        w = w / w.sum()
        w = 1.0 / w
        w = w / w.sum()
        self.register_buffer("weight", w)
        self.ce_kwargs = ce_kwargs

    def forward(self, logits, target):
        return nn.functional.cross_entropy(logits, target, weight=self.weight, **self.ce_kwargs)
