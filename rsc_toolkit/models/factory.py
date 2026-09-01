"""
모델 등록소(registry).

새 모델은 자기 모듈 안에서 데코레이터로 등록한다:

    from rsc_toolkit.models.factory import ModelFactory

    @ModelFactory.register_model_class("my_model")
    class MyModel(BaseRSCModel, nn.Module):
        ...

`import rsc_toolkit.models` 시점에 이 패키지 아래 모든 모듈을 import 하므로
(register_models) 모듈을 패키지 안에 두기만 하면 자동으로 등록된다.
설정 파일의 `model.name`(configs/model/*.yaml)은 여기 등록한 이름과 같아야 하고,
build_model 은 이 등록소에서 클래스를 찾는다.

등록 경로는 셋이고, 이 순서로 처리된다:

1. **패키지 내부 스캔** — `rsc_toolkit/models/` 안의 모듈은 자동으로 import 된다.
2. **엔트리포인트** (외부 패키지용). 자기 패키지 `pyproject.toml` 에 이렇게 적으면
   설치만으로 등록된다 — 툴킷을 포크할 필요가 없다:

       [project.entry-points."rsc_toolkit.models"]
       my_models = "my_pkg.models"

   값은 import 하면 데코레이터가 실행되는 모듈(또는 호출 가능 객체)을 가리키면 된다.
3. **`RSC_TOOLKIT_PLUGINS` 환경변수** — 아직 패키징하지 않은 로컬 모듈용 탈출구.
   쉼표로 구분한 모듈 경로를 import 한다:

       RSC_TOOLKIT_PLUGINS=my_models,lab.experiments.arch rsc-toolkit-train ...

모델이 지켜야 할 계약은 `rsc_toolkit/models/base.py` 의 `BaseRSCModel` 에 있다.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import pkgutil
from typing import Callable, Dict, List, Type

# 외부 패키지가 모델을 등록할 때 쓰는 엔트리포인트 그룹 이름.
PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.models"
# 패키징 없이 모듈을 붙일 때 쓰는 환경변수 (쉼표 구분).
PLUGIN_ENV_VAR = "RSC_TOOLKIT_PLUGINS"

logger = logging.getLogger(__name__)

# 이름 -> 모델 클래스. 모듈 간에 같은 객체를 공유한다.
_REGISTRY: Dict[str, type] = {}
_MODULES_IMPORTED = False

# 모델이 아니라서 스캔에서 빼는 모듈
_NON_MODEL_MODULES = {"base", "factory"}


class ModelFactory:
    @staticmethod
    def register_models() -> None:
        """
        rsc_toolkit.models 아래 모든 모듈을 import 해 데코레이터가 실행되게 한다. 멱등.

        선택 의존성이 없어 import 에 실패하는 모듈은 경고만 남기고 건너뛴다 —
        한 모델이 못 올라온다고 나머지까지 못 쓰게 되지 않도록.
        """
        global _MODULES_IMPORTED
        if _MODULES_IMPORTED:
            return
        # 아래 import 가 다시 이 함수를 부르는 재귀를 막기 위해 먼저 표시한다.
        _MODULES_IMPORTED = True

        import rsc_toolkit.models as package

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg or module_name.startswith("_") or module_name in _NON_MODEL_MODULES:
                continue  # backbone 같은 하위 패키지는 모델 모듈이 필요할 때 import 한다
            try:
                importlib.import_module(f"rsc_toolkit.models.{module_name}")
            except ImportError as exc:
                logger.warning(
                    "모델 모듈 rsc_toolkit.models.%s 을 불러오지 못해 등록에서 제외한다 "
                    "(선택 의존성 누락?): %s",
                    module_name,
                    exc,
                )

        ModelFactory._register_plugins()

    @staticmethod
    def _register_plugins() -> None:
        """
        패키지 밖에서 모델을 등록하는 두 경로를 처리한다 — 엔트리포인트와 환경변수.

        내부 스캔과 같은 원칙으로, **한 플러그인이 실패해도 나머지는 살린다**.
        (없으면 아무 일도 일어나지 않으므로 기존 동작에 영향이 없다.)
        """
        # ① 엔트리포인트. 3.9 는 dict, 3.10+ 는 select() 라 둘 다 받는다.
        try:
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, "select"):
                found = list(entry_points.select(group=PLUGIN_ENTRY_POINT_GROUP))
            else:  # pragma: no cover - Python 3.9
                found = list(entry_points.get(PLUGIN_ENTRY_POINT_GROUP, []))
        except Exception as exc:  # pragma: no cover - 메타데이터가 깨진 환경
            logger.warning("엔트리포인트 목록을 읽지 못했다: %s", exc)
            found = []

        for entry_point in found:
            try:
                entry_point.load()
                logger.info(
                    "모델 플러그인 등록: %s (엔트리포인트 %s)",
                    entry_point.value,
                    entry_point.name,
                )
            except Exception as exc:
                logger.warning(
                    "모델 플러그인 '%s' (%s) 을 불러오지 못해 건너뛴다: %s",
                    entry_point.name,
                    entry_point.value,
                    exc,
                )

        # ② 환경변수로 지정한 모듈.
        raw = os.environ.get(PLUGIN_ENV_VAR, "")
        for module_path in [part.strip() for part in raw.split(",") if part.strip()]:
            try:
                importlib.import_module(module_path)
                logger.info("모델 플러그인 등록: %s (%s)", module_path, PLUGIN_ENV_VAR)
            except Exception as exc:
                logger.warning(
                    "%s 가 가리키는 모듈 '%s' 을 불러오지 못해 건너뛴다: %s",
                    PLUGIN_ENV_VAR,
                    module_path,
                    exc,
                )

    @classmethod
    def register_model_class(cls, model_name: str) -> Callable[[type], type]:
        """클래스 데코레이터. 같은 이름에 다른 클래스를 다시 등록하면 실패한다."""
        name = str(model_name).strip()
        if not name:
            raise ValueError("모델 이름이 비어 있다")

        def decorator(model_cls: type) -> type:
            existing = _REGISTRY.get(name)
            if existing is not None and existing is not model_cls:
                raise ValueError(
                    f"모델 이름 '{name}' 은 이미 {existing.__module__}.{existing.__qualname__} 에 "
                    f"등록돼 있다. 다른 이름을 쓰거나 기존 등록을 정리하라."
                )
            _REGISTRY[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def registry(cls) -> Dict[str, type]:
        """이름 -> 클래스 딕셔너리 (살아 있는 객체. 이후 등록도 반영된다)."""
        cls.register_models()
        return _REGISTRY

    @classmethod
    def get_model_names(cls) -> List[str]:
        return sorted(cls.registry())

    @classmethod
    def get_model_class(cls, model_name: str) -> type:
        registry = cls.registry()
        if model_name not in registry:
            raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(registry)}")
        return registry[model_name]

    @classmethod
    def create_model(cls, model_name: str, **params):
        return cls.get_model_class(model_name)(**params)


def create_model(model_name: str, **params):
    """편의 함수. ModelFactory.create_model 과 같다."""
    return ModelFactory.create_model(model_name, **params)
