"""P2-6 — 스플릿 파서·인덱스 스키마의 단일 출처, 예전 이름들의 동일성, loaders.py deprecation."""
from __future__ import annotations

import warnings

import pytest

from rsc_toolkit.datasets.sample_index import CSV_COLUMNS, REQUIRED_INDEX_COLUMNS
from rsc_toolkit.datasets.splits import read_split_file


def test_read_split_file_semantics(tmp_path):
    f = tmp_path / "official_split.txt"
    f.write_text("# comment\n\n101_1b1_Al_sc_Meditron  TRAIN\n102_1b1_Ar_sc_Meditron test\nbroken\n")
    assert read_split_file(f) == {"101_1b1_Al_sc_Meditron": "train", "102_1b1_Ar_sc_Meditron": "test"}
    with pytest.raises(FileNotFoundError):
        read_split_file(tmp_path / "missing.txt")


def test_split_parser_is_shared_by_index_dataset_and_respirenet():
    from rsc_toolkit.datasets import respirenet_chunks
    from rsc_toolkit.datasets.index_dataset import IndexDataset

    assert respirenet_chunks.read_official_split is read_split_file
    assert "read_split_file" in IndexDataset._load_official_split_map.__doc__


def test_required_index_columns_single_source():
    from rsc_toolkit.datasets.base import BaseRSCDatasetProcessor

    assert REQUIRED_INDEX_COLUMNS == ("sample_id", "recording_path", "sample_level", "t0", "t1", "label", "meta")
    assert set(REQUIRED_INDEX_COLUMNS) == set(CSV_COLUMNS) - {"split"}
    assert BaseRSCDatasetProcessor.REQUIRED_INDEX_COLUMNS is REQUIRED_INDEX_COLUMNS


def test_former_collate_names_are_the_shared_implementations():
    from rsc_toolkit.datasets import bts_precomputed, btscard_sprs, collate, pcmcl_concat, qlung_data, respirenet_chunks, sprsound_ood
    from rsc_toolkit.trainer.data import index_datamodule

    assert index_datamodule._tuple_collate is collate.tuple_collate
    assert qlung_data.qlung_collate is collate.qlung_collate
    assert btscard_sprs.btscard_sprs_collate is collate.btscard_sprs_collate
    assert bts_precomputed.bts_collate is collate.stack_meta_collate
    assert sprsound_ood.sprsound_ood_collate is collate.stack_meta_collate
    assert respirenet_chunks.respirenet_collate is collate.index_meta_collate
    assert pcmcl_concat.pcmcl_train_collate is collate.pcmcl_train_collate


def test_loaders_module_is_deprecated():
    import importlib, sys

    sys.modules.pop("rsc_toolkit.datasets.loaders", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("rsc_toolkit.datasets.loaders")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
