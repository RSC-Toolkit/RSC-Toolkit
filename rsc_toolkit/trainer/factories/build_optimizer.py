from typing import Any, Dict

import torch.optim as optim

from rsc_toolkit.registries import OPTIMIZERS


def build_optimizer(cfg: Dict[str, Any], parameters) -> optim.Optimizer:
    """
    cfg example:
      {"name": "adam", "params": {"lr": 1e-3, "weight_decay": 0.0}}

    lr_scheduler 는 PLAdapter.configure_optimizers 가 처리한다 (여기 cfg 에는 없어야 한다).
    미지 name 은 가용 목록을 붙인 KeyError. params 는 복사해서 넘기므로 cfg 는 변하지 않는다.
    """
    return OPTIMIZERS.build(cfg, parameters)
