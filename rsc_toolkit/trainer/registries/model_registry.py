"""
모델 이름 -> 클래스 조회표.

등록의 원천은 rsc_toolkit.models.factory.ModelFactory 다 (각 모델 모듈의 데코레이터).
이 모듈은 build_model 이 써 온 `MODELS` 라는 이름을 유지하기 위한 얇은 창구일 뿐,
여기에 모델을 직접 적지 않는다. 새 모델은 rsc_toolkit/models/ 안에 모듈을 두고
데코레이터로 등록하면 된다.
"""
from typing import Dict, Type

import torch.nn as nn

from rsc_toolkit.models.factory import ModelFactory

# 같은 dict 객체를 공유하므로 이후 등록도 그대로 보인다.
MODELS: Dict[str, Type[nn.Module]] = ModelFactory.registry()
