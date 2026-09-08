from typing import Any, Dict

import torch.nn as nn

from rsc_toolkit.registries import MODELS


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
      {"name": "audioclap", "params": {"n_cls": 4}}

    미지 name 은 가용 목록을 붙인 KeyError. params 는 복사해서 넘기므로 cfg 는 변하지 않는다.
    """
    return MODELS.build(cfg)
