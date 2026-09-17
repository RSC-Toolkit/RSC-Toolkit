# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from platformdirs import user_data_dir


def default_datasets_base_dir(*, appname: str = "rsc_toolkit") -> Path:
    return (Path(user_data_dir(appname=appname, appauthor=False)) / "datasets").resolve()


def default_dataset_dir(dataset_name: str, *, appname: str = "rsc_toolkit") -> Path:
    return (default_datasets_base_dir(appname=appname) / str(dataset_name)).resolve()


def resolve_dataset_dir(
    dataset_name: str,
    *,
    dataset_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    appname: str = "rsc_toolkit",
) -> Path:
    if dataset_dir is None:
        return default_dataset_dir(dataset_name, appname=appname)

    p = Path(dataset_dir).expanduser()
    if not p.is_absolute():
        if project_root is None:
            raise ValueError("project_root is required for relative dataset_dir")
        p = Path(project_root).expanduser().resolve() / p
    return p.resolve()

