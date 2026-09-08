"""리팩토링 P1 8단계 — DatasetFactory → registries.DATASETS 위임, CLI 분리, preprocess --config_dir.

고정하는 것: 내장 데이터셋 스냅샷, DatasetFactory 공개 API 의 의미(중복 등록은 이제 ValueError, 미지 이름은
가용 목록 KeyError), CLI choices 가 registry 의 Mapping keys 라는 것, 외부 config_dir 로 data config 를 합성할 수
있다는 것, `rsc_toolkit.datasets` 패키지의 재수출(import 하위호환).
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

        DatasetFactory.register_dataset_class("_test_dup_ds")(First)  # 같은 클래스 재등록은 멱등
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
    """외부 디렉터리의 data/<이름>.yaml 을 --config_dir 로 합성할 수 있다 (학습 --config-dir 와 대칭)."""
    pkg_cfg = pathlib.Path(rsc_toolkit.__file__).resolve().parent / "configs" / "data" / "toy.yaml"
    ext = tmp_path / "cfgs" / "data"
    ext.mkdir(parents=True)
    text = pkg_cfg.read_text()
    (ext / "plugin_toy.yaml").write_text(text + "\n# external copy\n")
    cfg = datasets_cli._compose_data_config("plugin_toy", tmp_path, config_dir=str(tmp_path / "cfgs"))
    assert cfg["name"] == "toy" and cfg["params"]["project_root"] == str(tmp_path)

    # config_dir 없이 같은 이름을 부르면 패키지 목록에 없다고 안내한다
    with pytest.raises(ValueError, match="config_dir"):
        datasets_cli._compose_data_config("plugin_toy", tmp_path)
    with pytest.raises(FileNotFoundError):
        datasets_cli._compose_data_config("plugin_toy", tmp_path, config_dir=str(tmp_path / "missing"))
