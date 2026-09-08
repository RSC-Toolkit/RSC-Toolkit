"""
모델 등록소(registry) — 하위호환 창구.

등록의 원천은 이제 `rsc_toolkit.registries.MODELS` (공용 Registry) 다. 이 모듈은 기존 공개 API
(`ModelFactory.register_model_class` / `registry` / `get_model_names` / `get_model_class` /
`create_model` / `register_models`, 상수 `PLUGIN_ENTRY_POINT_GROUP` / `PLUGIN_ENV_VAR`)를 시그니처와
예외 의미 그대로 유지한다. 새 코드는 둘 중 어느 쪽을 써도 같은 등록소에 닿는다:

    from rsc_toolkit.models.factory import ModelFactory
    @ModelFactory.register_model_class("my_model")
    class MyModel(BaseRSCModel, nn.Module): ...

    from rsc_toolkit.registries import MODELS          # 같은 것
    @MODELS.register("my_model")
    class MyModel(BaseRSCModel, nn.Module): ...

`import rsc_toolkit.models` 시점에 이 패키지 아래 모든 모듈을 import 하므로(register_models)
모듈을 패키지 안에 두기만 하면 자동으로 등록된다. 설정 파일의 `model.name`(configs/model/*.yaml)은
여기 등록한 이름과 같아야 하고, build_model 은 이 등록소에서 클래스를 찾는다.

등록 경로는 셋이고, 이 순서로 처리된다:

1. **패키지 내부 스캔** — `rsc_toolkit/models/` 안의 모듈은 자동으로 import 된다.
2. **엔트리포인트** (외부 패키지용). 자기 패키지 `pyproject.toml` 에 이렇게 적으면
   설치만으로 등록된다 — 툴킷을 포크할 필요가 없다:

       [project.entry-points."rsc_toolkit.plugins"]     # 모든 컴포넌트 종류 공용 그룹
       my_models = "my_pkg.models"

   예전 그룹 `rsc_toolkit.models` 도 계속 읽는다. 값은 import 하면 데코레이터가 실행되는
   모듈(또는 호출 가능 객체)을 가리키면 된다.
3. **`RSC_TOOLKIT_PLUGINS` 환경변수** — 아직 패키징하지 않은 로컬 모듈용 탈출구.
   쉼표로 구분한 모듈 경로를 import 한다:

       RSC_TOOLKIT_PLUGINS=my_models,lab.experiments.arch rsc-toolkit-train ...

모델이 지켜야 할 계약은 `rsc_toolkit/models/base.py` 의 `BaseRSCModel` 에 있다.
"""
from __future__ import annotations

from typing import Callable, List

from rsc_toolkit.registries import MODELS
from rsc_toolkit.registry import PLUGIN_ENV_VAR, Registry, load_plugins

# 외부 패키지가 모델을 등록할 때 써 온 엔트리포인트 그룹 이름 (하위호환; 공용 그룹은
# rsc_toolkit.registry.PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.plugins").
PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.models"

__all__ = ["ModelFactory", "create_model", "PLUGIN_ENTRY_POINT_GROUP", "PLUGIN_ENV_VAR"]


class ModelFactory:
    @staticmethod
    def register_models() -> None:
        """
        rsc_toolkit.models 아래 모든 모듈을 import 해 데코레이터가 실행되게 하고 플러그인을 읽는다. 멱등.

        선택 의존성이 없어 import 에 실패하는 모듈은 경고만 남기고 건너뛴다 —
        한 모델이 못 올라온다고 나머지까지 못 쓰게 되지 않도록.
        """
        MODELS._discover()

    @staticmethod
    def _register_plugins() -> None:
        """엔트리포인트·환경변수 플러그인을 (다시) 읽는다. 한 플러그인이 실패해도 나머지는 살린다."""
        load_plugins(force=True)

    @classmethod
    def register_model_class(cls, model_name: str) -> Callable[[type], type]:
        """클래스 데코레이터. 같은 이름에 다른 클래스를 다시 등록하면 실패한다."""
        return MODELS.register(model_name)

    @classmethod
    def registry(cls) -> Registry:
        """이름 -> 클래스 등록소 (살아 있는 객체. 이후 등록도 반영된다). Mapping + pop 을 지원한다."""
        MODELS._discover()
        return MODELS

    @classmethod
    def get_model_names(cls) -> List[str]:
        return sorted(cls.registry())

    @classmethod
    def get_model_class(cls, model_name: str) -> type:
        return cls.registry().get(model_name)  # 미지 이름: KeyError("Unknown model '...'. Available: [...]")

    @classmethod
    def create_model(cls, model_name: str, **params):
        return cls.get_model_class(model_name)(**params)


def create_model(model_name: str, **params):
    """편의 함수. ModelFactory.create_model 과 같다."""
    return ModelFactory.create_model(model_name, **params)
