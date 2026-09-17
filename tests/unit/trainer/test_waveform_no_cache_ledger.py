"""Processors that do not use the cache (kind=waveform) neither check nor write the chain ledger.

Previously IndexDataModule also checked `_chain.json` in the shared cache directory for waveform, so when it disagreed with a ledger
left by another waveform config (different desired_length_sec / resample_after_slice), training that never reads the cache died with a RuntimeError.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from rsc_toolkit.datasets.feature_cache import FeatureCache
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule


class _Rec:
    def __init__(self, **kw):
        self.kwargs = kw
        self.df = None

    def __len__(self):
        return 0


def _dm(tmp_path, kind, **extra):
    (tmp_path / "index").mkdir(exist_ok=True)
    (tmp_path / "index" / "index_cycle.csv").write_text("sample_id\n")
    return IndexDataModule(dataset_name="icbhi", index_dataset_cls=f"{__name__}._Rec", project_root=str(tmp_path),
                           data_dir=str(tmp_path), sampling_rate=16000, features={"enabled": True, "kind": kind}, **extra)


def test_waveform_skips_chain_ledger_even_when_it_conflicts(tmp_path):
    # set up a case where a different preprocessing chain is already recorded in the same cache directory under the same key
    dm_other = _dm(tmp_path, "waveform", desired_length_sec=9.0, resample_after_slice=True)
    key_cache = FeatureCache(cache_dir=tmp_path / "features_cache")
    key_cache.configure({"kind": "waveform", "sampling_rate": 16000}, processor_name="WaveformProcessor",
                        audio_chain=dm_other._audio_chain())
    ledger = key_cache.features_dir / "_chain.json"
    assert ledger.exists()
    before = ledger.read_text()

    dm = _dm(tmp_path, "waveform", desired_length_sec=10.0)  # different chain — previously a RuntimeError here
    fp = dm._make_feature_processor_with_cache()
    assert fp.feature_cache is None
    assert ledger.read_text() == before  # neither checked nor written


def test_cached_kind_still_guards_the_chain(tmp_path):
    dm_a = _dm(tmp_path, "fbank", desired_length_sec=8.0)
    dm_a._make_feature_processor_with_cache()
    dm_b = _dm(tmp_path, "fbank", desired_length_sec=9.0)
    with pytest.raises(RuntimeError, match="audio chain mismatch"):
        dm_b._make_feature_processor_with_cache()
