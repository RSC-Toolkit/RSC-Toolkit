"""rsc_toolkit.datasets — raw 데이터셋(다운로드·인덱스·검증)과 학습용 index dataset/datamodule 들.

등록소는 `DatasetFactory`(= registries.DATASETS 위임). CLI(`rsctk datasets …`)는 `rsc_toolkit.datasets.cli` 에
있고, 여기서는 예전 import 경로(`from rsc_toolkit.datasets import build_parser, handle_command, …`)를 위해
재수출만 한다. 새 코드는 `rsc_toolkit.datasets.cli` 를 직접 쓰라 (monkeypatch 대상도 그쪽이다).
"""
from rsc_toolkit.datasets.factory import DatasetFactory

DatasetFactory.register_datasets()

from rsc_toolkit.datasets.cli import (  # noqa: E402  (등록 뒤에 import — CLI choices 가 등록소를 읽는다)
    _available_data_configs,
    _compose_data_config,
    _configure_processor_raw_data_directory,
    _default_project_root,
    _find_project_root,
    _handle_preprocess,
    _resolve_under_root,
    build_parser,
    create_dataset,
    handle_command,
)

__all__ = [
    "DatasetFactory",
    "build_parser",
    "handle_command",
    "create_dataset",
    "_find_project_root",
    "_default_project_root",
    "_resolve_under_root",
    "_available_data_configs",
    "_compose_data_config",
    "_handle_preprocess",
    "_configure_processor_raw_data_directory",
]
