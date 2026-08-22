"""
학습 엔트리포인트의 재현성 관련 동작 테스트.

- config 의 seed 가 실제로 적용되는지 (이전에는 아무도 읽지 않아 무효였다)
- train.py 의 타입 어노테이션에 미정의 이름이 없는지
- EMA 콜백이 config 그룹으로 붙고 beta 가 정상값인지
"""
import logging
import typing

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

import rsc_toolkit.train as train_mod


CONFIG_PATH = "../../rsc_toolkit/configs"


def _compose(name: str, overrides=None):
    with initialize(version_base=None, config_path=CONFIG_PATH):
        return compose(config_name=name, overrides=overrides or [])


def test_seed_is_applied_and_makes_rng_reproducible():
    """_apply_seed 를 부르면 이후 난수열이 결정된다."""
    train_mod._apply_seed({"seed": 123})
    first = torch.randn(8)

    train_mod._apply_seed({"seed": 123})
    second = torch.randn(8)

    assert torch.equal(first, second)


def test_different_seeds_give_different_rng():
    train_mod._apply_seed({"seed": 123})
    a = torch.randn(8)

    train_mod._apply_seed({"seed": 456})
    b = torch.randn(8)

    assert not torch.equal(a, b)


def test_missing_seed_warns(caplog):
    """seed 가 없으면 조용히 넘어가지 말고 경고한다."""
    with caplog.at_level(logging.WARNING):
        train_mod._apply_seed({})

    assert any("seed" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "func",
    [train_mod._get_loggers, train_mod._log_run_metadata, train_mod._apply_seed],
)
def test_annotations_resolve(func):
    """
    어노테이션의 이름이 전부 실제로 정의돼 있는지 확인한다.

    train.py 는 `from __future__ import annotations` 를 쓰므로 미정의 이름이
    있어도 import 시점에는 조용하다. 이전에 lightning 을 import 하지 않은 채
    `L.Trainer` 를 어노테이션으로 쓰고 있었고, get_type_hints 에서 NameError 가 났다.
    """
    typing.get_type_hints(func)


@pytest.mark.parametrize("config_name", ["default", "default_ast"])
def test_ema_callback_is_configured_with_sane_beta(config_name):
    """
    EMA 콜백이 실제로 붙고 beta 가 정상값인지 확인한다.

    이전에는 trainer/default.yaml 에 beta: 0.5 가 인라인으로 박혀 있었다.
    0.5 는 반감기가 1 스텝이라 사실상 EMA 가 아니다. 한편 callbacks/ema.yaml
    (0.999) 은 어느 defaults 에도 연결돼 있지 않아 쓰이지 않았다.
    """
    cfg = _compose(config_name)
    callbacks = OmegaConf.to_container(cfg.trainer.params.callbacks, resolve=True)

    ema = [c for c in callbacks if "EMACallback" in c.get("_target_", "")]
    assert len(ema) == 1, f"EMA 콜백이 정확히 하나여야 한다: {callbacks}"
    assert ema[0]["beta"] == 0.999


@pytest.mark.parametrize("config_name", ["default", "default_ast"])
def test_callbacks_can_be_disabled(config_name):
    """`callbacks=none` 으로 콜백을 끌 수 있어야 한다."""
    cfg = _compose(config_name, overrides=["callbacks=none"])
    assert list(cfg.trainer.params.callbacks) == []


def test_callbacks_group_does_not_leak_top_level_key():
    """
    callbacks 그룹이 `# @package _global_` 없이 로드되면 cfg.callbacks 아래에
    통째로 들어가고 trainer 에는 반영되지 않는다. 그 상태를 막는다.
    """
    cfg = _compose("default")
    assert "callbacks" not in cfg
