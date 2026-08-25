from typing import Dict, Type
import torch.nn as nn

from rsc_toolkit.trainer.losses import WeightedCrossEntropyFromCounts, WeightedCrossEntropyFromPrior

LOSSES: Dict[str, Type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "cross_entropy_from_counts": WeightedCrossEntropyFromCounts,
    "cross_entropy_from_prior": WeightedCrossEntropyFromPrior,
}
