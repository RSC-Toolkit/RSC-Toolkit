"""
모델 구현 모음.

모델은 각 모듈에서 `@ModelFactory.register_model_class("<이름>")` 로 등록하고,
이 패키지를 import 하는 순간 모든 모듈이 import 되어 등록이 끝난다.
조회는 ModelFactory / create_model / build_model(trainer) 로 한다.
"""
from rsc_toolkit.models.factory import ModelFactory, create_model

ModelFactory.register_models()

__all__ = ["ModelFactory", "create_model"]
