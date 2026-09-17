"""Unknown-key error in build_datamodule — when index-family keys injected by `data_defaults@data: base` reach a strict datamodule,
it raises a TypeError that names the cause (injected defaults, removal override)."""
from __future__ import annotations

import pytest

from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule, unknown_datamodule_kwargs


class _Strict:
    def __init__(self, path, batch_size=8):
        self.path, self.batch_size = path, batch_size


class _Loose:
    def __init__(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs


def test_unknown_kwargs_detection():
    assert unknown_datamodule_kwargs(_Strict, {"path": "x", "batch_size": 4}) == []
    assert unknown_datamodule_kwargs(_Strict, {"path": "x", "pad_type": "repeat", "typo": 1}) == ["pad_type", "typo"]
    assert unknown_datamodule_kwargs(_Loose, {"path": "x", "anything": 1}) == []


def test_build_datamodule_names_the_injected_defaults(monkeypatch):
    from rsc_toolkit.registries import DATAMODULES

    monkeypatch.setitem(DATAMODULES._objs, "_strict_test", _Strict)
    cfg = {"name": "strict", "params": {"datamodule_name": "_strict_test", "path": "x",
                                        "desired_length_sec": 8.0, "pad_type": "repeat", "specaugment": {"enabled": False}}}
    with pytest.raises(TypeError) as exc:
        build_datamodule(cfg)
    msg = str(exc.value)
    assert "_strict_test" in msg and "data_defaults@data: base" in msg
    assert "~data.params.desired_length_sec" in msg and "~data.params.specaugment" in msg
    # cfg is not mutated
    assert cfg["params"]["datamodule_name"] == "_strict_test"


def test_build_datamodule_typo_is_reported_without_defaults_hint(monkeypatch):
    from rsc_toolkit.registries import DATAMODULES

    monkeypatch.setitem(DATAMODULES._objs, "_strict_test", _Strict)
    with pytest.raises(TypeError) as exc:
        build_datamodule({"params": {"datamodule_name": "_strict_test", "path": "x", "batch_sizee": 4}})
    assert "['batch_sizee']" in str(exc.value) and "data_defaults" not in str(exc.value)


def test_real_combination_bts_precomputed_with_base_defaults():
    """Combination found in practice: compose('default', ['data=bts_precomputed']) — previously a TypeError with no cause."""
    pytest.importorskip("hydra")
    import pathlib

    import rsc_toolkit
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    cfg_dir = pathlib.Path(rsc_toolkit.__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="default", overrides=["data=bts_precomputed"])
    data = OmegaConf.to_container(cfg.data, resolve=False)
    with pytest.raises(TypeError) as exc:
        build_datamodule(data)
    assert "bts_precomputed" in str(exc.value) and "~data.params.pad_type" in str(exc.value)
