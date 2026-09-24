"""
Integration tests for `rsctk datasets preprocess`.

This command exists to "fill the feature cache before training", so
the key check is that the training path (IndexDataModule) reads the cache filled by the CLI
back with the same directory, key and sample_id (0 cache misses).

The previous implementation was broken in two ways:
- the CLI had no way to build feature_processor, so it always raised RuntimeError
- the preprocess_features_from_index it called was commented out
"""
import argparse
import pathlib

import pandas as pd
import pytest

from rsc_toolkit.datasets import cli as datasets_cli  # monkeypatch target is the CLI module
from rsc_toolkit.datasets.toy_dataset import ToyDataset


def _repo_root() -> pathlib.Path:
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("repository root not found (no pyproject.toml)")


REPO_ROOT = _repo_root()


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        command="datasets",
        dataset_command="preprocess",
        dataset="toy",
        data_config="toy_fbank",  # fbank: computed quickly without downloading external models
        project_root=str(REPO_ROOT),
        data_dir=None,
        sample_level=None,
        workers=None,
        batch_size=None,
        cache_dir=None,
        force=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _load_split_map() -> dict:
    split_path = REPO_ROOT / "sample_data" / "toy_dataset" / "official_split.txt"
    mapping = {}
    for line in split_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1].lower()
    return mapping


@pytest.fixture
def toy_data_dir(tmp_path) -> pathlib.Path:
    """
    Builds an index in tmp from the bundled toy raw data and, for speed,
    keeps only the cycles of 2 train + 1 test recordings.
    """
    data_dir = tmp_path / "toy"
    proc = ToyDataset.make_processor(outputs_directory=str(data_dir), force=False)
    proc.build_index(sample_level="cycle")

    index_csv = data_dir / "index" / "index_cycle.csv"
    df = pd.read_csv(index_csv)

    split_map = _load_split_map()
    stems = df["recording_path"].map(lambda p: pathlib.Path(str(p)).stem)
    train_recs = sorted({s for s in stems if split_map.get(s) == "train"})[:2]
    test_recs = sorted({s for s in stems if split_map.get(s) == "test"})[:1]
    keep = set(train_recs) | set(test_recs)
    assert train_recs and test_recs, "fixture precondition: both splits must have recordings"

    df[stems.isin(keep)].to_csv(index_csv, index=False)
    return data_dir


def _training_datamodule(data_dir: pathlib.Path, cache_dir: pathlib.Path):
    """Builds the datamodule exactly as training does, without going through the CLI."""
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule

    cfg = datasets_cli._compose_data_config("toy_fbank", REPO_ROOT)
    cfg["params"]["data_dir"] = str(data_dir)
    cfg["params"]["features"]["cache_dir"] = str(cache_dir)
    cfg["params"]["feature_precompute"] = False
    dm = build_datamodule(cfg)
    dm.setup("fit")
    return dm


def _assert_all_cached(dm) -> int:
    from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

    total = 0
    for name, ds in (("train", dm.train_ds), ("val", dm.val_ds)):
        sample_ids = IndexDataModule._collect_sample_ids(ds.index_ds)
        assert sample_ids, f"{name} split is empty"
        missing = [s for s in sample_ids if not ds.fp.feature_cache.is_cached(s)]
        assert missing == [], f"{name} split cache misses: {missing[:5]}..."
        total += len(sample_ids)
    return total


def test_preprocess_fills_cache_that_training_reuses(toy_data_dir, tmp_path):
    """
    Key check: the training path reads the cache filled by the CLI without cache misses.

    Any mismatch in the cache directory or key convention shows up as misses here.
    """
    cache_dir = tmp_path / "fc"

    datasets_cli.handle_command(
        _make_args(data_dir=str(toy_data_dir), cache_dir=str(cache_dir))
    )

    dm = _training_datamodule(toy_data_dir, cache_dir)
    total = _assert_all_cached(dm)

    stats = dm.train_ds.fp.feature_cache.get_cache_stats()
    assert stats["pt_files"] == total  # all of train+val(=test) in one key directory


def test_preprocess_force_recomputes(toy_data_dir, tmp_path):
    """--force clears the existing cache for the same settings and refills everything."""
    cache_dir = tmp_path / "fc"
    datasets_cli.handle_command(
        _make_args(data_dir=str(toy_data_dir), cache_dir=str(cache_dir))
    )

    # corrupt one cache file beforehand to show that force recomputation actually happened
    features_dirs = list((cache_dir / "features").iterdir())
    assert len(features_dirs) == 1
    pt_files = sorted(features_dirs[0].glob("*.pt"))
    corrupted = pt_files[0]
    corrupted.write_bytes(b"broken")

    datasets_cli.handle_command(
        _make_args(data_dir=str(toy_data_dir), cache_dir=str(cache_dir), force=True)
    )

    # the corrupted file must have been recomputed and restored to a valid tensor
    import torch

    tensor = torch.load(corrupted, map_location="cpu")
    assert torch.is_tensor(tensor)

    dm = _training_datamodule(toy_data_dir, cache_dir)
    _assert_all_cached(dm)


def test_preprocess_rejects_mismatched_data_config(toy_data_dir, tmp_path):
    """Rejects a positional dataset that differs from dataset_name in the config."""
    with pytest.raises(ValueError, match="dataset_name"):
        datasets_cli.handle_command(
            _make_args(
                dataset="icbhi",
                data_config="toy_fbank",
                data_dir=str(toy_data_dir),
                cache_dir=str(tmp_path / "fc"),
            )
        )


def test_preprocess_unknown_data_config_lists_available(tmp_path):
    with pytest.raises(ValueError, match="toy_fbank"):
        datasets_cli.handle_command(
            _make_args(data_config="does_not_exist", data_dir=str(tmp_path))
        )


def test_preprocess_missing_index_points_to_build_index(tmp_path):
    """Without an index it fails and points to build-index."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="build-index"):
        datasets_cli.handle_command(
            _make_args(data_dir=str(empty), cache_dir=str(tmp_path / "fc"))
        )


def test_preprocess_rejects_config_without_features(toy_data_dir, tmp_path, monkeypatch):
    """A config with features disabled has nothing to extract, so it is rejected explicitly."""
    real_compose = datasets_cli._compose_data_config

    def fake_compose(name, project_root):
        cfg = real_compose(name, project_root)
        cfg["params"]["features"]["enabled"] = False
        return cfg

    monkeypatch.setattr(datasets_cli, "_compose_data_config", fake_compose)

    with pytest.raises(ValueError, match="features"):
        datasets_cli.handle_command(
            _make_args(data_dir=str(toy_data_dir), cache_dir=str(tmp_path / "fc"))
        )
