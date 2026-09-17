"""Parity after moving to the shared registry — every existing config must resolve to the same class.

Pinned:
- built-in list snapshot (no name from the former manual dicts is missing)
- configs/{loss,optimizer,metrics,data}/*.yaml and the adapter of each top-level config all resolve from the registry
- non-mutation and error-message contract of the build_* factories, no silent fallback in build_adapter, model-contract check for split_iter
- importing data_registry does not eagerly import the paper datamodules (optional-dependency isolation)
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import torch
import yaml
from torch import nn

import rsc_toolkit
from rsc_toolkit.registries import ADAPTERS, DATAMODULES, LOSSES, METRICS, MODELS, OPTIMIZERS
from rsc_toolkit.trainer.factories.build_adapter import build_adapter
from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
from rsc_toolkit.trainer.factories.build_loss import build_loss
from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
from rsc_toolkit.trainer.factories.build_optimizer import build_optimizer

CONFIGS = pathlib.Path(rsc_toolkit.__file__).resolve().parent / "configs"


def _yaml(path: pathlib.Path):
    return yaml.safe_load(path.read_text()) or {}


# ── built-in list snapshot (guards against omissions when moving manual dicts to the lazy hub) ─────────────
def test_builtin_snapshot_matches_former_manual_dicts():
    assert set(DATAMODULES) == {
        "index", "bts_precomputed", "respirenet", "pcmcl", "amsd", "gapaug",
        "lungmix", "protonet", "domaintransfer", "qlung", "btscard_sprs",
    }
    assert set(LOSSES) == {"cross_entropy", "cross_entropy_from_counts", "cross_entropy_from_prior", "bce_with_logits"}
    assert set(METRICS) == {"topk_accuracy", "icbhi_score", "icbhi_paper_binary"}
    assert set(OPTIMIZERS) == {"adam", "adamw", "sgd"}
    assert set(ADAPTERS) == {"pl", "split_iter"}


def test_legacy_registry_modules_expose_the_same_objects():
    from rsc_toolkit.trainer.registries import (
        data_registry, loss_registry, metric_registry, model_registry, optimizer_registry,
    )

    assert data_registry.DATAMODULES is DATAMODULES
    assert loss_registry.LOSSES is LOSSES
    assert metric_registry.METRICS is METRICS
    assert optimizer_registry.OPTIMIZERS is OPTIMIZERS
    assert model_registry.MODELS is MODELS


def test_data_registry_import_is_lazy():
    """data_registry import used to pull in all 10 paper datamodules — one missing librosa
    blocked all training. Now there are only declarations, and the import happens on first lookup."""
    code = (
        "import sys, rsc_toolkit.trainer.registries.data_registry as r; "
        "assert 'respirenet' in r.DATAMODULES; "
        "leaked = [m for m in sys.modules if m.startswith('rsc_toolkit.datasets.') and m.endswith(('respirenet_chunks','pcmcl_concat','qlung_data','amsd_data'))]; "
        "assert not leaked, leaked; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


# ── resolve every config ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", sorted((CONFIGS / "loss").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_loss_config_builds(path):
    cfg = _yaml(path)
    assert cfg["name"] in LOSSES
    assert isinstance(build_loss(cfg), nn.Module)


@pytest.mark.parametrize("path", sorted((CONFIGS / "optimizer").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_optimizer_config_builds(path):
    cfg = _yaml(path)
    assert cfg["name"] in OPTIMIZERS
    assert issubclass(OPTIMIZERS.get(cfg["name"]), torch.optim.Optimizer)
    # lr in the yaml can be a ${...} interpolation (resolved by hydra), so only name resolution is checked here.
    opt = build_optimizer({"name": cfg["name"], "params": {"lr": 0.01}}, [nn.Parameter(torch.zeros(1))])
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.parametrize("path", sorted((CONFIGS / "metrics").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_metrics_config_builds(path):
    cfg = _yaml(path)
    train_fns, val_fns = build_train_val_metric_factories(cfg)
    for item in (cfg.get("train") or []) + (cfg.get("val") or []):
        assert item["name"] in METRICS
    for fn in train_fns + val_fns:
        assert callable(fn)


@pytest.mark.parametrize("path", sorted((CONFIGS / "data").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_data_config_names_a_registered_datamodule(path):
    cfg = _yaml(path)
    name = (cfg.get("params") or {}).get("datamodule_name") or cfg.get("datamodule")
    assert name in DATAMODULES, f"{path.name}: datamodule '{name}' not registered (registered: {sorted(DATAMODULES)})"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_top_level_config_adapter_is_registered(path):
    cfg = _yaml(path)
    adapter = cfg.get("adapter")
    if not adapter:
        return
    name = adapter.get("name") if isinstance(adapter, dict) else adapter
    assert name in ADAPTERS, f"{path.name}: adapter '{name}' not registered"


# ── build_* contract ───────────────────────────────────────────────────────────────────
def test_build_loss_and_model_do_not_mutate_cfg_and_report_unknown_names():
    cfg = {"name": "cross_entropy", "params": {"label_smoothing": 0.1}}
    loss = build_loss(cfg)
    assert isinstance(loss, nn.CrossEntropyLoss) and loss.label_smoothing == 0.1
    assert cfg == {"name": "cross_entropy", "params": {"label_smoothing": 0.1}}
    with pytest.raises(KeyError, match="Unknown loss '_nope'. Available"):
        build_loss({"name": "_nope"})
    with pytest.raises(KeyError, match="Unknown metric '_nope'. Available"):
        build_train_val_metric_factories({"val": [{"name": "_nope"}]})
    with pytest.raises(KeyError, match="has no 'name'"):
        build_train_val_metric_factories({"val": [{"params": {}}]})


class _RecordingDM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_datamodule_is_non_destructive_and_names_the_failing_key():
    DATAMODULES.register("_test_recording_dm")(_RecordingDM)
    try:
        cfg = {"name": "icbhi", "params": {"datamodule_name": "_test_recording_dm", "batch_size": 8}}
        dm = build_datamodule(cfg)
        assert isinstance(dm, _RecordingDM) and dm.kwargs == {"batch_size": 8}
        assert cfg["params"] == {"datamodule_name": "_test_recording_dm", "batch_size": 8}  # not popped
        build_datamodule(cfg)  # reusable (previously the second call raised KeyError)

        # a top-level datamodule key is accepted too
        dm2 = build_datamodule({"name": "icbhi", "datamodule": "_test_recording_dm", "params": {"x": 1}})
        assert dm2.kwargs == {"x": 1}
    finally:
        DATAMODULES.pop("_test_recording_dm", None)

    with pytest.raises(KeyError, match="datamodule_name"):
        build_datamodule({"name": "icbhi", "params": {"batch_size": 8}})
    # names the failing key exactly (previously it printed cfg["name"]='icbhi', which was misleading)
    with pytest.raises(KeyError, match="Unknown datamodule '_no_such_dm' \\(data.params.datamodule_name\\). Available"):
        build_datamodule({"name": "icbhi", "params": {"datamodule_name": "_no_such_dm"}})


class _PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.head(audio_batch)
        return logits, loss_func(logits, labels)


class _SplitIterModel(_PlainModel):
    def train_transform(self, x):
        return x

    def val_transform(self, x):
        return x

    def forward(self, x):
        return self.head(x)


_OPT = {"name": "sgd", "params": {"lr": 0.1}}


def test_build_adapter_default_and_named():
    from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter
    from rsc_toolkit.trainer.adapters.split_iter_adapter import SplitIterAdapter

    ce = nn.CrossEntropyLoss()
    a = build_adapter(None, rsc_model=_PlainModel(), loss_func=ce, optimizer_cfg=_OPT)
    assert type(a) is PLAdapter
    a = build_adapter({}, rsc_model=_PlainModel(), loss_func=ce, optimizer_cfg=_OPT)
    assert type(a) is PLAdapter
    b = build_adapter(
        {"name": "split_iter", "params": {"split_iter": 3}},
        rsc_model=_SplitIterModel(), loss_func=ce, optimizer_cfg=_OPT,
    )
    assert isinstance(b, SplitIterAdapter) and b.K == 3 and b.automatic_optimization is False


def test_build_adapter_rejects_typos_instead_of_silently_falling_back():
    with pytest.raises(KeyError, match="Unknown adapter 'split-iter'. Available: \\['pl', 'split_iter'\\]"):
        build_adapter({"name": "split-iter"}, rsc_model=_PlainModel(), loss_func=nn.CrossEntropyLoss(), optimizer_cfg=_OPT)


def test_split_iter_requires_the_m2d_protocol_at_construction():
    with pytest.raises(TypeError, match="train_transform"):
        build_adapter(
            {"name": "split_iter", "params": {"split_iter": 4}},
            rsc_model=_PlainModel(), loss_func=nn.CrossEntropyLoss(), optimizer_cfg=_OPT,
        )
    with pytest.raises(ValueError, match="split_iter"):
        build_adapter(
            {"name": "split_iter", "params": {"split_iter": 0}},
            rsc_model=_SplitIterModel(), loss_func=nn.CrossEntropyLoss(), optimizer_cfg=_OPT,
        )
