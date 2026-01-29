from typing import Dict, Type
import torch.nn as nn

LOSSES: Dict[str, Type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
}
