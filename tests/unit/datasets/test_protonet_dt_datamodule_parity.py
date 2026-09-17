"""Parity after removing the `__new__` swap in the ProtoNet and Domain Transfer datamodules.

The golden `tests/golden/protonet_dt_datamodule_fingerprint.json` is a fingerprint recorded on real data (ICBHI) with the **former implementation**
(swapping IndexDataset for Proto/DTIndexDataset after setup): dataset class, length, dedicated attributes, waveform sums of the first 3 samples,
splice count, last (splice) sample, balanced class weight. The new implementation (injection at construction + `_index_dataset_kwargs`/role) must produce the same values.
Skipped in environments without the ICBHI index (CI).
"""
from __future__ import annotations

import json
import pathlib
import random

import numpy as np
import pytest
import torch

import rsc_toolkit

REPO = pathlib.Path(rsc_toolkit.__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden" / "protonet_dt_datamodule_fingerprint.json"
needs_icbhi = pytest.mark.skipif(not (REPO / ".datasets/icbhi/index/index_cycle.csv").exists(), reason="ICBHI index missing")
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")

ATTR_KEYS = ("cycle_len", "is_train", "bandpass", "target_len", "aug_prob", "use_noise", "use_stretch", "use_pitch",
             "snr_low", "snr_high", "stretch_range", "pitch_semitones", "stretch_impl", "fade", "sampling_rate")


def _data_cfg(name):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
        cfg = compose(config_name="default", overrides=[f"data={name}"])
    cfg.data.params.project_root = str(REPO)
    d = OmegaConf.to_container(cfg.data, resolve=True)
    d["params"]["feature_precompute"] = False
    d["params"]["num_workers"] = 0
    return d


def _fp_items(base, n=3, seed=0):
    out = []
    for i in range(min(n, len(base))):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        it = base[i]
        a = it["audio"].float().flatten()
        out.append({"sample_id": str(it["sample_id"]), "label": it.get("label"), "shape": list(it["audio"].shape),
                    "sum": round(float(a.sum()), 4), "abs_sum": round(float(a.abs().sum()), 4), "n": int(a.numel())})
    return out


def fingerprint(name):
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule

    dm = build_datamodule(_data_cfg(name))
    dm.setup(None)
    rec = {"config": name, "class": type(dm).__name__, "class_weight": getattr(dm, "train_class_weight", None)}
    for role in ("train_ds", "val_ds", "test_ds"):
        ds = getattr(dm, role)
        base = getattr(ds, "index_ds", ds)
        r = {"wrapper": type(ds).__name__, "base": type(base).__name__, "len": len(base),
             "attrs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__)
                       for k, v in vars(base).items() if k in ATTR_KEYS},
             "n_extra": len(getattr(base, "extra", []) or []),
             "items_eval": _fp_items(base, 3)}
        if role == "train_ds" and getattr(base, "extra", None):
            e = base.extra[0]
            r["extra0"] = {"label": int(e["label"]), "keys": sorted(e.keys())}
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            it = base[len(base) - 1]
            a = it["audio"].float().flatten()
            r["splice_last"] = {"label": it.get("label"), "n": int(a.numel()), "sum": round(float(a.sum()), 4)}
        rec[role] = r
    return rec


@needs_icbhi
@needs_golden
@pytest.mark.parametrize("name", ["icbhi_protonet", "icbhi_domaintransfer"])
def test_datamodule_fingerprint_matches_former_swap_implementation(name):
    golden = json.loads(GOLDEN.read_text())[name]
    assert "error" not in golden, golden
    golden.pop("seconds", None)
    now = fingerprint(name)
    assert now == golden


def test_former_swap_helpers_are_gone():
    from rsc_toolkit.datasets.domaintransfer_data import DomainTransferDataModule, DTIndexDataset
    from rsc_toolkit.datasets.protonet_data import ProtoIndexDataset, ProtoNetDataModule

    assert "_swap_datasets" not in vars(ProtoNetDataModule)
    assert "_index_dataset_kwargs" in vars(ProtoNetDataModule) and "_index_dataset_kwargs" in vars(DomainTransferDataModule)
    # dedicated argument assembly: is_train depends on role, and DT passes the augmentation arguments through
    p = ProtoNetDataModule.__new__(ProtoNetDataModule)
    p.index_csv, p.raw_data_directory, p.sampling_rate, p.sample_level, p.audio_preprocessor = "i.csv", "/raw", 4000, "cycle", None
    p.fade, p.resample_after_slice, p.time_unit, p.split_file, p.index_dataset_params = True, False, "s", None, {}
    p.cycle_len, p.bandpass = 16000, True
    assert ProtoNetDataModule._index_dataset_kwargs(p, split="train", role="train")["is_train"] is True
    assert ProtoNetDataModule._index_dataset_kwargs(p, split="test", role="val")["is_train"] is False
    d = DomainTransferDataModule.__new__(DomainTransferDataModule)
    d.index_csv, d.raw_data_directory, d.sampling_rate, d.sample_level, d.audio_preprocessor = "i.csv", "/raw", 8000, "cycle", None
    d.fade, d.resample_after_slice, d.time_unit, d.split_file, d.index_dataset_params = True, False, "s", None, {}
    d.target_len, d.aug_kwargs = 64000, dict(aug_prob=0.5, use_noise=True, use_stretch=True, use_pitch=True, stretch_impl="linear")
    kw = DomainTransferDataModule._index_dataset_kwargs(d, split="train", role="train")
    assert kw["is_train"] is True and kw["target_len"] == 64000 and kw["stretch_impl"] == "linear" and kw["aug_prob"] == 0.5
    assert issubclass(DTIndexDataset, object) and issubclass(ProtoIndexDataset, object)
