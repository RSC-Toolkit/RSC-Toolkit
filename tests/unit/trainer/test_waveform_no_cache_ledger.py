"""캐시를 쓰지 않는 프로세서(kind=waveform)는 체인 대장을 대조·기록하지 않는다 (09-01 리뷰 F 오탐, 2026-09-08 GaP-Aug 재발).

예전엔 IndexDataModule 이 waveform 에도 공용 캐시 디렉터리의 `_chain.json` 을 대조해, 다른 waveform config(다른
desired_length_sec·resample_after_slice)가 남긴 대장과 어긋나면 캐시를 읽지도 않는 학습이 RuntimeError 로 죽었다.
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
    # 같은 캐시 디렉터리·같은 키에 다른 전처리 체인이 이미 기록돼 있는 상황을 만든다
    dm_other = _dm(tmp_path, "waveform", desired_length_sec=9.0, resample_after_slice=True)
    key_cache = FeatureCache(cache_dir=tmp_path / "features_cache")
    key_cache.configure({"kind": "waveform", "sampling_rate": 16000}, processor_name="WaveformProcessor",
                        audio_chain=dm_other._audio_chain())
    ledger = key_cache.features_dir / "_chain.json"
    assert ledger.exists()
    before = ledger.read_text()

    dm = _dm(tmp_path, "waveform", desired_length_sec=10.0)  # 체인이 다르다 — 예전엔 여기서 RuntimeError
    fp = dm._make_feature_processor_with_cache()
    assert fp.feature_cache is None
    assert ledger.read_text() == before  # 대조도, 기록도 하지 않았다


def test_cached_kind_still_guards_the_chain(tmp_path):
    dm_a = _dm(tmp_path, "fbank", desired_length_sec=8.0)
    dm_a._make_feature_processor_with_cache()
    dm_b = _dm(tmp_path, "fbank", desired_length_sec=9.0)
    with pytest.raises(RuntimeError, match="체인 불일치"):
        dm_b._make_feature_processor_with_cache()
