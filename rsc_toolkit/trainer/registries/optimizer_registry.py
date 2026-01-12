from typing import Dict, Type
import torch.optim as optim

OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}
