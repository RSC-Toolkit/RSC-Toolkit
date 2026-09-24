"""DatasetFactory -> registries.DATASETS delegation, CLI split, preprocess --config_dir.

Pinned: the built-in dataset snapshot, the semantics of the DatasetFactory public API (duplicate registration now raises ValueError,
an unknown name raises KeyError listing the available ones), CLI choices being the registry Mapping keys, composing a data config
from an external config_dir, and the re-exports of the `rsc_toolkit.datasets` package (import backward compatibility).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import pytest

import rsc_toolkit
import rsc_toolkit.datasets as datasets_pkg
from rsc_toolkit.datasets import cli as datasets_cli
from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.datasets.factory import DatasetFactory
from rsc_toolkit.registries import DATASETS


def test_builtin_datasets_snapshot_and_delegation():
    assert {"icbhi", "sprsound", "hflungv1", "toy"} <= set(DATASETS)
    assert DatasetFactory.datasets is DATASETS
    assert set(DatasetFactory.get_dataset_names()) == set(DATASETS)
    from rsc_toolkit.datasets.icbhi.icbhi_dataset import ICBHIDataset

    assert DatasetFactory.get_dataset_class("icbhi") is ICBHIDataset
    assert issubclass(DatasetFactory.get_dataset_class("toy"), ICBHIDataset)


def test_unknown_dataset_lists_available():
    with pytest.raises(KeyError, match="Unknown dataset '_nope'. Available"):
        DatasetFactory.get_dataset_class("_nope")


def test_duplicate_registration_is_rejected_instead_of_silently_overwritten():
    @DatasetFactory.register_dataset_class("_test_dup_ds")
    class First(BaseRSCDataset):
        pass

    try:
        with pytest.raises(ValueError, match="_test_dup_ds"):

            @DatasetFactory.register_dataset_class("_test_dup_ds")
            class Second(BaseRSCDataset):
                pass

        DatasetFactory.register_dataset_class("_test_dup_ds")(First)  # re-registering the same class is idempotent
        assert DatasetFactory.get_dataset_class("_test_dup_ds") is First
    finally:
        DATASETS.pop("_test_dup_ds", None)


def test_cli_choices_come_from_the_registry():
    @DATASETS.register("_test_plugin_ds")
    class PluginDS(BaseRSCDataset):
        pass

    try:
        parser = argparse.ArgumentParser()
        datasets_cli.build_parser(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["datasets", "validate", "_test_plugin_ds"])
        assert args.dataset == "_test_plugin_ds"
    finally:
        DATASETS.pop("_test_plugin_ds", None)


def test_package_reexports_cli_names_for_backward_compatibility():
    for name in ("build_parser", "handle_command", "create_dataset", "_find_project_root", "_resolve_under_root",
                 "_compose_data_config", "DatasetFactory"):
        assert getattr(datasets_pkg, name) is getattr(datasets_cli, name) or name == "DatasetFactory"


def test_compose_data_config_with_external_config_dir(tmp_path):
    """data/<name>.yaml in an external directory can be composed via --config_dir (symmetric with training --config-dir)."""
    pkg_cfg = pathlib.Path(rsc_toolkit.__file__).resolve().parent / "configs" / "data" / "toy.yaml"
    ext = tmp_path / "cfgs" / "data"
    ext.mkdir(parents=True)
    text = pkg_cfg.read_text()
    (ext / "plugin_toy.yaml").write_text(text + "\n# external copy\n")
    cfg = datasets_cli._compose_data_config("plugin_toy", tmp_path, config_dir=str(tmp_path / "cfgs"))
    assert cfg["name"] == "toy" and cfg["params"]["project_root"] == str(tmp_path)

    # calling the same name without config_dir reports that it is not in the package list
    with pytest.raises(ValueError, match="config_dir"):
        datasets_cli._compose_data_config("plugin_toy", tmp_path)
    with pytest.raises(FileNotFoundError):
        datasets_cli._compose_data_config("plugin_toy", tmp_path, config_dir=str(tmp_path / "missing"))
