# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""rsc_toolkit.datasets -- raw datasets (download, index, validation) and index datasets/datamodules for training.

The registry is `DatasetFactory` (delegates to registries.DATASETS). The CLI (`rsctk datasets ...`) lives in
`rsc_toolkit.datasets.cli`; this module only re-exports it for the legacy import path
(`from rsc_toolkit.datasets import build_parser, handle_command, ...`). New code should use `rsc_toolkit.datasets.cli`
directly (it is also the monkeypatch target).
"""
from rsc_toolkit.datasets.factory import DatasetFactory

DatasetFactory.register_datasets()

from rsc_toolkit.datasets.cli import (  # noqa: E402  (imported after registration -- CLI choices read the registry)
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
