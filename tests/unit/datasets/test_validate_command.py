"""
Tests for the verdict/exit-code behavior of `rsctk datasets validate`.

Previously the return value of the validation function was discarded and "✅ passed" was always printed,
so even a directory with 0 wav files was reported as passing.
"""
import argparse
import pathlib

import pytest

from rsc_toolkit.datasets import cli as datasets_cli  # monkeypatch target is the CLI module


class _Downloader:
    def __init__(self, ok):
        self._ok = ok

    def validate_downloaded_files(self):
        return self._ok


class _Processor:
    """The CLI injects the path only if raw_data_directory exists."""

    def __init__(self, result):
        self._result = result
        self.raw_data_directory = None

    def validate_processed_files(self):
        return self._result


class _Dataset:
    def __init__(self, downloaded_ok, processed_result):
        self.downloader = _Downloader(downloaded_ok)
        self.processor = _Processor(processed_result)


def _run_validate(monkeypatch, tmp_path, *, downloaded_ok, processed_result):
    monkeypatch.setattr(
        datasets_cli,
        "create_dataset",
        lambda args: _Dataset(downloaded_ok, processed_result),
    )
    args = argparse.Namespace(
        command="datasets",
        dataset_command="validate",
        dataset="icbhi",
        data_dir=str(tmp_path),
        project_root=None,
        force=False,
    )
    return datasets_cli.handle_command(args)


def test_validate_fails_when_download_invalid(monkeypatch, tmp_path):
    """If download validation is False it must fail (non-zero exit code)."""
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=False, processed_result=None
    )
    assert rc != 0


def test_validate_passes_when_download_valid(monkeypatch, tmp_path):
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=True, processed_result=True
    )
    assert rc == 0


def test_unimplemented_processed_check_is_not_a_failure(monkeypatch, tmp_path):
    """
    validate_processed_files returning None (not implemented) does not count as a failure.

    The icbhi/sprsound/hflungv1 processors are all unimplemented stubs now,
    so treating it as a failure would make every valid dataset fail.
    """
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=True, processed_result=None
    )
    assert rc == 0


def test_validate_fails_when_processed_invalid(monkeypatch, tmp_path):
    """Unlike not implemented (None), an explicit False counts as a failure."""
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=True, processed_result=False
    )
    assert rc != 0


@pytest.mark.parametrize("dataset_name", ["icbhi", "sprsound", "hflungv1"])
def test_processors_report_none_when_nothing_processed(dataset_name, tmp_path):
    """
    An empty output directory where build-index has not run yet must give None (nothing to check).

    Returning False would be misreported as a 'validation failure' by the logic above.
    """
    from rsc_toolkit.datasets.factory import DatasetFactory

    cls = DatasetFactory.get_dataset_class(dataset_name=dataset_name)
    processor = cls.make_processor(outputs_directory=str(tmp_path), force=False)
    assert processor.validate_processed_files() is None


# ---------------------------------------------------------------------------
# Actual validation of index outputs (base default implementation + dataset hooks)
# ---------------------------------------------------------------------------

def _repo_root():
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


@pytest.fixture
def icbhi_processor_with_index(tmp_path):
    """ICBHI processor with a real cycle index built from the bundled toy raw data (ICBHI format)."""
    from rsc_toolkit.datasets.icbhi.processor import ICBHIDatasetProcessor

    proc = ICBHIDatasetProcessor(
        outputs_directory=str(tmp_path),
        raw_data_directory=str(_repo_root() / "sample_data" / "toy_dataset" / "files"),
    )
    proc.build_index(sample_level="cycle")
    return proc


def test_real_index_validates_true(icbhi_processor_with_index):
    assert icbhi_processor_with_index.validate_processed_files() is True


def test_index_with_zero_rows_is_invalid(icbhi_processor_with_index, tmp_path):
    """
    Even if build-index exits 0, training is impossible with 0 rows.
    (This actually happened with SPRSound.)
    """
    index_csv = tmp_path / "index" / "index_cycle.csv"
    header = index_csv.read_text().splitlines()[0]
    index_csv.write_text(header + "\n")
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_index_missing_required_column_is_invalid(icbhi_processor_with_index, tmp_path):
    import pandas as pd

    index_csv = tmp_path / "index" / "index_cycle.csv"
    df = pd.read_csv(index_csv).drop(columns=["label"])
    df.to_csv(index_csv, index=False)
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_stale_stats_file_is_invalid(icbhi_processor_with_index, tmp_path):
    """If total_samples in stats differs from the row count, the index and stats are out of sync."""
    import json

    stats = tmp_path / "index" / "index_cycle_stats.json"
    data = json.loads(stats.read_text())
    data["total_samples"] = 1
    stats.write_text(json.dumps(data))
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_only_existing_levels_are_checked(icbhi_processor_with_index):
    """Building only cycle and not clip is not a failure (only existing levels are checked)."""
    assert icbhi_processor_with_index.validate_processed_files() is True


def test_cycle_index_with_missing_labels_is_invalid(icbhi_processor_with_index, tmp_path):
    import pandas as pd

    index_csv = tmp_path / "index" / "index_cycle.csv"
    df = pd.read_csv(index_csv)
    df.loc[0, "label"] = None
    df.to_csv(index_csv, index=False)
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_sprsound_requires_official_split_next_to_index(tmp_path):
    """Per-dataset auxiliary output hook: sprsound needs official_split.txt next to the index."""
    from rsc_toolkit.datasets.sprsound.processor import SPRSoundDatasetProcessor

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_cycle.csv").write_text(
        "sample_id,recording_path,sample_level,t0,t1,split,label,meta\n"
        "a_0.0_1.0,a.wav,cycle,0.0,1.0,train,0,{}\n"
    )
    proc = SPRSoundDatasetProcessor(outputs_directory=str(tmp_path))

    assert proc.validate_processed_files() is False  # auxiliary file missing

    (index_dir / "official_split.txt").write_text("a train\n")
    assert proc.validate_processed_files() is True


def test_feature_cache_validate_entries_is_shape_agnostic(tmp_path):
    """
    Feature cache validation does not assume a shape — it differs per model/feature kind.
    Only existence and loadability are checked.
    """
    import torch
    from rsc_toolkit.datasets.feature_cache import FeatureCache

    cache = FeatureCache(cache_dir=tmp_path)
    cache.configure(processor_config={"kind": "any"}, processor_name="AnyProcessor")
    cache.put("clap_like", torch.randn(1, 512))
    cache.put("fbank_like", torch.randn(1024, 128))
    cache.put("broken", torch.randn(3))
    cache._get_cache_path("broken").write_bytes(b"not a tensor")

    report = cache.validate_entries(["clap_like", "fbank_like", "broken", "never_cached"])
    assert report["total"] == 4
    assert report["missing"] == ["never_cached"]
    assert report["corrupted"] == ["broken"]
    assert report["ok"] is False

    assert cache.validate_entries(["clap_like", "fbank_like"])["ok"] is True
