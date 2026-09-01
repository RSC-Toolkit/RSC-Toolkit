"""
서드파티 확장점 회귀 테스트.

툴킷의 최종 목표는 "외부 연구자가 설치 후 자기 모델만 구현하면 학습·검증이 도는 것" 이다.
그 목표를 지탱하는 것이 ① BaseRSCModel 계약 ② 세 가지 등록 경로 두 가지이므로,
둘 다 여기서 고정한다. (2026-09-01 신설 — 그전에는 외부 확장 경로에 테스트가 없었다.)
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
    """등록소는 프로세스 전역이라 테스트가 서로 오염되지 않게 되돌린다."""
    ModelFactory.registry().pop(name, None)


class TestContract:
    def test_training_forward_is_required(self):
        """계약을 안 지킨 모델은 인스턴스화 자체가 막혀야 한다 — 학습 중에 터지면 늦다."""

        class Incomplete(BaseRSCModel, nn.Module):
            pass

        with pytest.raises(TypeError, match="training_forward"):
            Incomplete()

    def test_minimal_conforming_model_runs(self):
        """README 에 적은 최소 예제가 실제로 도는지 — 문서와 코드가 어긋나지 않도록."""

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
        meta 를 안 쓰는 모델도 **kwargs 를 열어야 한다 —
        PLAdapter 가 meta_label_mode 데이터셋에서 meta_labels 를 넘기기 때문이다.
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
            meta_labels=torch.tensor([0, 1]),  # 넘겨도 죽지 않아야 한다
        )
        assert logits.shape == (2, 4)


class TestRegistration:
    def test_core_models_still_register(self):
        """확장점을 추가하면서 내부 스캔이 깨지지 않았는지."""
        registry = ModelFactory.registry()
        assert len(registry) >= 30
        for name in ("ast", "beats", "resnet18_sft"):
            assert name in registry, f"핵심 모델 {name} 이 등록되지 않았다"
        assert all(issubclass(cls, BaseRSCModel) for cls in registry.values())

    def test_env_var_plugin_path(self, tmp_path, monkeypatch):
        """RSC_TOOLKIT_PLUGINS 로 지정한 외부 모듈이 등록된다 (패키징 없이)."""
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
        플러그인 하나가 깨져도 나머지 모델은 살아 있어야 한다 —
        남의 코드 때문에 툴킷 전체가 못 쓰게 되면 안 된다.
        """
        monkeypatch.setenv(PLUGIN_ENV_VAR, "rsc_definitely_not_a_real_module")
        ModelFactory._register_plugins()  # 예외를 밖으로 던지지 않아야 한다
        assert "ast" in ModelFactory.registry()

    def test_entry_point_group_is_stable(self):
        """
        외부 패키지가 pyproject 에 적는 그룹 이름이라 함부로 바꾸면 남의 설치가 조용히 깨진다.
        README 와 factory 독스트링에 적힌 값과 같아야 한다.
        """
        assert PLUGIN_ENTRY_POINT_GROUP == "rsc_toolkit.models"
        assert PLUGIN_ENV_VAR == "RSC_TOOLKIT_PLUGINS"
