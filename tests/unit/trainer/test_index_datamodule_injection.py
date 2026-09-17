"""IndexDataModule: 3 extension points (additive) + split_role + subclass hooks.

With defaults the code path must equal the former one (constructor arguments, collate, wrapper); given values, they are injected without a fork.
QLung and Lungmix moved to the `_index_dataset_kwargs` hook, removing the copy-pasted kwargs block and the identity comparison — this checks that the
parent rules (fade/resample_after_slice/time_unit/split_file) are inherited automatically.
"""
from __future__ import annotations

import operator
import pathlib

import pytest
import torch

from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule, _tuple_collate


class _RecordingIndexDataset:
    """Stand-in for index_dataset_cls — records only the constructor arguments."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = None
        _RecordingIndexDataset.instances.append(self)

    def __len__(self):
        return 0


def _make_dm(tmp_path: pathlib.Path, **extra) -> IndexDataModule:
    (tmp_path / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "index" / "index_cycle.csv").write_text("sample_id\n")
    return IndexDataModule(
        dataset_name="dummy",
        index_dataset_cls=f"{__name__}._RecordingIndexDataset",
        project_root=str(tmp_path),
        data_dir=str(tmp_path),
        sampling_rate=16000,
        **extra,
    )


def test_defaults_reproduce_the_former_kwargs_exactly(tmp_path):
    dm = _make_dm(tmp_path)
    ds = dm._build_index_dataset_single(split="train")
    assert set(ds.kwargs) == {"index_csv", "data_root", "sampling_rate", "sample_level", "split", "audio_preprocessor"}
    assert ds.kwargs["split"] == "train" and ds.kwargs["sampling_rate"] == 16000
    assert dm.collate_fn is _tuple_collate and dm.dataset_wrapper_cls is None and dm.index_dataset_params == {}


def test_non_default_audio_options_are_still_forwarded_conditionally(tmp_path):
    dm = _make_dm(tmp_path, fade=False, resample_after_slice=True, time_unit="ms_int")
    kw = dm._index_dataset_kwargs(split="test")
    assert kw["fade"] is False and kw["resample_after_slice"] is True and kw["time_unit"] == "ms_int"


def test_index_dataset_params_are_passed_through_and_win(tmp_path):
    dm = _make_dm(tmp_path, index_dataset_params={"bandpass_hz": [25, 400], "fade": False})
    ds = dm._build_index_dataset_single(split="train")
    assert ds.kwargs["bandpass_hz"] == [25, 400]
    assert ds.kwargs["fade"] is False  # explicit disable beats the parent rule (omitted when True)


def test_collate_fn_accepts_dotted_path_or_callable(tmp_path):
    dm = _make_dm(tmp_path, collate_fn="operator.itemgetter")
    assert dm.collate_fn is operator.itemgetter
    dm.train_ds = [1, 2, 3]
    assert dm.train_dataloader().collate_fn is operator.itemgetter

    def my_collate(b):
        return b

    dm2 = _make_dm(tmp_path, collate_fn=my_collate)
    assert dm2.collate_fn is my_collate


class _Wrapper(IndexFeatureDataset):
    pass


def test_dataset_wrapper_cls_replaces_index_feature_dataset(tmp_path):
    dm = _make_dm(
        tmp_path,
        features={"enabled": True, "kind": "waveform"},
        dataset_wrapper_cls=f"{__name__}._Wrapper",
    )
    base = _RecordingIndexDataset()
    wrapped = dm._wrap_with_features(base, split_role="train")
    assert type(wrapped) is _Wrapper and wrapped.index_ds is base

    plain = _make_dm(tmp_path, features={"enabled": True, "kind": "waveform"})
    assert type(plain._wrap_with_features(base, split_role="val")) is IndexFeatureDataset


def test_setup_passes_split_roles(tmp_path, monkeypatch):
    dm = _make_dm(tmp_path, features={"enabled": True, "kind": "waveform"})
    roles = []

    def spy(base_ds, *, feature_transform=None, meta_label_fn=None, split_role=None):
        roles.append(split_role)
        return base_ds

    monkeypatch.setattr(dm, "_wrap_with_features", spy)
    dm.setup(None)
    assert roles == ["train", "val", "test"]


def test_subclass_hooks_inherit_parent_rules():
    """QLung/Lungmix _index_dataset_kwargs add only their own arguments on top of the parent rules (evidence the copy-paste is gone)."""
    from rsc_toolkit.datasets.lungmix_data import LungmixDataModule
    from rsc_toolkit.datasets.qlung_data import QLungDataModule

    assert "_build_index_dataset_single" not in LungmixDataModule.__dict__
    assert "_build_index_dataset_single" not in QLungDataModule.__dict__
    for name in ("train_dataloader", "val_dataloader", "test_dataloader", "_loader"):
        assert name not in QLungDataModule.__dict__

    fake = LungmixDataModule.__new__(LungmixDataModule)  # attributes only, no __init__ (super() needs an instance)
    fake.index_csv, fake.raw_data_directory, fake.sampling_rate = "i.csv", "/raw", 16000
    fake.sample_level, fake.audio_preprocessor = "cycle", None
    fake.fade, fake.resample_after_slice, fake.time_unit, fake.split_file = False, False, "ms_int", None
    fake.index_dataset_params = {}
    fake.lungmix_bandpass, fake.lungmix_label_map = {"low": 50}, "sprsound_7to4"
    kw = LungmixDataModule._index_dataset_kwargs(fake, split="train")
    assert kw["bandpass"] == {"low": 50} and kw["label_map"] == "sprsound_7to4"
    assert kw["fade"] is False and kw["time_unit"] == "ms_int" and "resample_after_slice" not in kw
