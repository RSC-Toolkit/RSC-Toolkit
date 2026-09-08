"""하위호환 창구. 등록의 원천은 rsc_toolkit.registries.MODELS (각 모델 모듈의 데코레이터) 다.

build_model 이 써 온 `MODELS` 라는 이름을 유지할 뿐, 여기에 모델을 직접 적지 않는다.
새 모델은 rsc_toolkit/models/ 안에 모듈을 두고 데코레이터로 등록하면 된다.
"""
from rsc_toolkit.registries import MODELS

__all__ = ["MODELS"]
