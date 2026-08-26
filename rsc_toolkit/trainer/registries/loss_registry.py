from typing import Dict, Type
import torch.nn as nn

from rsc_toolkit.trainer.losses import WeightedCrossEntropyFromCounts, WeightedCrossEntropyFromPrior

LOSSES: Dict[str, Type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "cross_entropy_from_counts": WeightedCrossEntropyFromCounts,
    "cross_entropy_from_prior": WeightedCrossEntropyFromPrior,
    "bce_with_logits": nn.BCEWithLogitsLoss,      # PC-MCL 3-label 주 손실 (wa976/PC-MCL main.py:421)
}
