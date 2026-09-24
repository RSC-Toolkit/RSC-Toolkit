"""
Regression tests for third-party extension points.

The toolkit's goal is that "an external researcher installs it, implements only their own model, and training/validation run".
That goal rests on (1) the BaseRSCModel contract and (2) the three registration paths,
so both are pinned here.
"""
from __future__ import annotations

import sys
import textwrap

import pytest
import torch
from torch import nn

from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import (
    PLUGIN_ENTRY_POINT_GROUP,
    PLUGIN_ENV_VAR,
    ModelFactory,
)


def _unregister(name: str) -> None:
    """The registry is process-global, so restore it to keep tests from polluting each other."""
    ModelFactory.registry().pop(name, None)


class TestContract:
    def test_training_forward_is_required(self):
        """A model that breaks the contract must be blocked at instantiation — failing during training is too late."""

        class Incomplete(BaseRSCModel, nn.Module):
            pass

        with pytest.raises(TypeError, match="training_forward"):
            Incomplete()

    def test_minimal_conforming_model_runs(self):
        """The minimal example in the README actually runs — so docs and code do not diverge."""

        class Minimal(BaseRSCModel, nn.Module):
            def __init__(self, n_cls: int = 4):
                super().__init__()
                self.head = nn.Linear(16, n_cls)

            def training_forward(self, audio_batch, labels, loss_func, **kwargs):
                logits = self.head(audio_batch.mean(dim=-1))
                return logits, loss_func(logits, labels)

        model = Minimal(n_cls=4)
        logits, loss = model.training_forward(
            audio_batch=torch.randn(3, 16, 8),
            labels=torch.tensor([0, 1, 2]),
            loss_func=nn.CrossEntropyLoss(),
        )
        assert logits.shape == (3, 4)
        assert loss.requires_grad and loss.ndim == 0

    def test_kwargs_are_open_for_meta_labels(self):
        """
        Models that do not use meta must still accept **kwargs —
        PLAdapter passes meta_labels for meta_label_mode datasets.
        """

        class IgnoresMeta(BaseRSCModel, nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(16, 4)

            def training_forward(self, audio_batch, labels, loss_func, **kwargs):
                logits = self.head(audio_batch.mean(dim=-1))
                return logits, loss_func(logits, labels)

        logits, _ = IgnoresMeta().training_forward(
            audio_batch=torch.randn(2, 16, 8),
            labels=torch.tensor([0, 1]),
            loss_func=nn.CrossEntropyLoss(),
            meta_labels=torch.tensor([0, 1]),  # passing it must not crash
        )
        assert logits.shape == (2, 4)


class TestRegistration:
    def test_core_models_still_register(self):
        """Adding extension points did not break the internal scan."""
        registry = ModelFactory.registry()
        assert len(registry) >= 30
        for name in ("ast", "beats", "resnet18_sft"):
            assert name in registry, f"core model {name} is not registered"
        assert all(issubclass(cls, BaseRSCModel) for cls in registry.values())

    def test_env_var_plugin_path(self, tmp_path, monkeypatch):
        """An external module named in RSC_TOOLKIT_PLUGINS gets registered (without packaging)."""
        module_name = "rsc_env_plugin_fixture"
        (tmp_path / f"{module_name}.py").write_text(
            textwrap.dedent(
                """
                from torch import nn
                from rsc_toolkit.models.base import BaseRSCModel
                from rsc_toolkit.models.factory import ModelFactory

                @ModelFactory.register_model_class("env_plugin_fixture")
                class EnvPluginFixture(BaseRSCModel, nn.Module):
                    def __init__(self, **params):
                        super().__init__()
                        self.head = nn.Linear(4, 4)

                    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
                        logits = self.head(audio_batch)
                        return logits, loss_func(logits, labels)
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv(PLUGIN_ENV_VAR, module_name)
        try:
            ModelFactory._register_plugins()
            assert "env_plugin_fixture" in ModelFactory.registry()
            assert ModelFactory.create_model("env_plugin_fixture") is not None
        finally:
            _unregister("env_plugin_fixture")
            sys.modules.pop(module_name, None)

    def test_broken_plugin_does_not_kill_the_registry(self, monkeypatch):
        """
        If one plugin is broken the other models must stay alive —
        someone else's code must not make the whole toolkit unusable.
        """
        monkeypatch.setenv(PLUGIN_ENV_VAR, "rsc_definitely_not_a_real_module")
        ModelFactory._register_plugins()  # must not raise
        assert "ast" in ModelFactory.registry()

    def test_entry_point_group_is_stable(self):
        """
        This group name is written in external packages' pyproject, so changing it carelessly silently breaks their installs.
        It must equal the value documented in the README and the factory docstring.
        """
        assert PLUGIN_ENTRY_POINT_GROUP == "rsc_toolkit.models"
        assert PLUGIN_ENV_VAR == "RSC_TOOLKIT_PLUGINS"
