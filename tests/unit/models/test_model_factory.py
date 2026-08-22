"""
ModelFactory 가 유일한 모델 등록소로 동작하는지 확인한다.

이전에는 register_models() 가 "_model" 이름 규칙의 모듈만 찾아 아무것도 등록하지
못했고(MODELS == {}), 실제 조회는 trainer/registries/model_registry.py 의 정적 dict 가
따로 맡고 있었다. 이제 모델 모듈의 데코레이터가 원천이고 model_registry.MODELS 는
같은 dict 를 가리킨다. 다른 사람이 모델을 추가할 때 모듈만 두면 되게 하기 위함이다.
"""
import pathlib

import pytest
import torch.nn as nn
import yaml

import rsc_toolkit.models  # noqa: F401  (import 가 등록을 트리거한다)
from rsc_toolkit.models.factory import ModelFactory, create_model
from rsc_toolkit.trainer.factories.build_model import build_model
from rsc_toolkit.trainer.registries.model_registry import MODELS

BUILTIN = {"ast", "audioclap", "cnn", "efficientnet", "resnet"}
CONFIG_DIR = pathlib.Path(rsc_toolkit.models.__file__).resolve().parents[1] / "configs" / "model"


def test_builtin_models_are_registered():
    assert BUILTIN <= set(ModelFactory.get_model_names())


def test_model_registry_is_the_same_object_as_factory_registry():
    """build_model 이 읽는 MODELS 가 별도 사본이 아니라 등록소 그 자체여야 한다."""
    assert MODELS is ModelFactory.registry()


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_model_config_name_is_registered(config_path):
    """configs/model/*.yaml 의 name 이 등록소에 없으면 학습이 KeyError 로 죽는다."""
    cfg = yaml.safe_load(config_path.read_text())
    name = cfg["name"]
    assert name in MODELS, f"{config_path.name}: model.name={name!r} 이 등록되지 않았다. 등록: {sorted(MODELS)}"


def test_decorator_registers_new_model_and_build_model_finds_it():
    """새 모델은 데코레이터 한 줄로 build_model 까지 연결돼야 한다."""

    @ModelFactory.register_model_class("_test_dummy_model")
    class Dummy(nn.Module):
        def __init__(self, n_cls: int):
            super().__init__()
            self.head = nn.Linear(4, n_cls)

    try:
        built = build_model({"name": "_test_dummy_model", "params": {"n_cls": 3}})
        assert isinstance(built, Dummy)
        assert built.head.out_features == 3
        assert isinstance(create_model("_test_dummy_model", n_cls=2), Dummy)
    finally:
        ModelFactory.registry().pop("_test_dummy_model", None)


def test_duplicate_name_with_different_class_is_rejected():
    """같은 이름에 다른 클래스를 조용히 덮어쓰면 어느 모델이 도는지 알 수 없게 된다."""

    @ModelFactory.register_model_class("_test_dup")
    class First(nn.Module):
        pass

    try:
        with pytest.raises(ValueError, match="_test_dup"):

            @ModelFactory.register_model_class("_test_dup")
            class Second(nn.Module):
                pass

        # 같은 클래스를 다시 등록하는 것(모듈 재로드 등)은 허용
        ModelFactory.register_model_class("_test_dup")(First)
    finally:
        ModelFactory.registry().pop("_test_dup", None)


def test_unknown_model_error_lists_available():
    with pytest.raises(KeyError, match="Available"):
        ModelFactory.get_model_class("_no_such_model")
    with pytest.raises(KeyError, match="Available"):
        build_model({"name": "_no_such_model", "params": {}})


def test_register_models_is_idempotent():
    before = dict(ModelFactory.registry())
    ModelFactory.register_models()
    ModelFactory.register_models()
    assert ModelFactory.registry() == before
