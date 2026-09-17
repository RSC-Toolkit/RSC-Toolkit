"""
Checks that ModelFactory acts as the single model registry.

Previously register_models() only searched modules following a "_model" naming rule and registered nothing
(MODELS == {}), while lookups were handled separately by a static dict in trainer/registries/model_registry.py.
Now the decorators in the model modules are the source, and model_registry.MODELS points to
the same dict, so adding a model only requires adding its module.
"""
import pathlib

import pytest
import torch.nn as nn
import yaml

import rsc_toolkit.models  # noqa: F401  (the import triggers registration)
from rsc_toolkit.models.factory import ModelFactory, create_model
from rsc_toolkit.trainer.factories.build_model import build_model
from rsc_toolkit.trainer.registries.model_registry import MODELS

BUILTIN = {"ast", "audioclap", "cnn", "efficientnet", "resnet"}
CONFIG_DIR = pathlib.Path(rsc_toolkit.models.__file__).resolve().parents[1] / "configs" / "model"


def test_builtin_models_are_registered():
    assert BUILTIN <= set(ModelFactory.get_model_names())


def test_model_registry_is_the_same_object_as_factory_registry():
    """The MODELS read by build_model must be the registry itself, not a separate copy."""
    assert MODELS is ModelFactory.registry()


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_model_config_name_is_registered(config_path):
    """If a name in configs/model/*.yaml is not registered, training dies with KeyError."""
    cfg = yaml.safe_load(config_path.read_text())
    name = cfg["name"]
    assert name in MODELS, f"{config_path.name}: model.name={name!r} is not registered. Registered: {sorted(MODELS)}"


def test_decorator_registers_new_model_and_build_model_finds_it():
    """A new model must be wired through to build_model with a single decorator line."""

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
    """Silently overwriting a name with a different class makes it impossible to know which model runs."""

    @ModelFactory.register_model_class("_test_dup")
    class First(nn.Module):
        pass

    try:
        with pytest.raises(ValueError, match="_test_dup"):

            @ModelFactory.register_model_class("_test_dup")
            class Second(nn.Module):
                pass

        # re-registering the same class (e.g. module reload) is allowed
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
