"""
Tests for reproducibility-related behavior of the training entry point.

- the seed in the config is actually applied (previously nothing read it, so it had no effect)
- the type annotations in train.py contain no undefined names
- the EMA callback is attached via a config group and beta has a sane value
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
    """Calling _apply_seed makes the subsequent random sequence deterministic."""
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
    """A missing seed triggers a warning instead of being silently skipped."""
    with caplog.at_level(logging.WARNING):
        train_mod._apply_seed({})

    assert any("seed" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "func",
    [train_mod._get_loggers, train_mod._log_run_metadata, train_mod._apply_seed],
)
def test_annotations_resolve(func):
    """
    Checks that every name in the annotations is actually defined.

    train.py uses `from __future__ import annotations`, so undefined names
    stay silent at import time. `L.Trainer` was once used as an annotation
    without importing lightning, and get_type_hints raised NameError.
    """
    typing.get_type_hints(func)


@pytest.mark.parametrize("config_name", ["default", "default_ast"])
def test_ema_callback_is_configured_with_sane_beta(config_name):
    """
    Checks that the EMA callback is actually attached and beta has a sane value.

    trainer/default.yaml used to hard-code beta: 0.5 inline.
    0.5 has a half-life of 1 step, so it is effectively not an EMA. Meanwhile callbacks/ema.yaml
    (0.999) was not wired into any defaults and went unused.
    """
    cfg = _compose(config_name)
    callbacks = OmegaConf.to_container(cfg.trainer.params.callbacks, resolve=True)

    ema = [c for c in callbacks if "EMACallback" in c.get("_target_", "")]
    assert len(ema) == 1, f"there must be exactly one EMA callback: {callbacks}"
    assert ema[0]["beta"] == 0.999


@pytest.mark.parametrize("config_name", ["default", "default_ast"])
def test_callbacks_can_be_disabled(config_name):
    """Callbacks can be disabled with `callbacks=none`."""
    cfg = _compose(config_name, overrides=["callbacks=none"])
    assert list(cfg.trainer.params.callbacks) == []


def test_callbacks_group_does_not_leak_top_level_key():
    """
    If the callbacks group is loaded without `# @package _global_`, it lands entirely under
    cfg.callbacks and is not applied to the trainer. This guards against that.
    """
    cfg = _compose("default")
    assert "callbacks" not in cfg
