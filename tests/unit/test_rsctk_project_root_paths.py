# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from pathlib import Path

from rsc_toolkit.datasets import _find_project_root, _resolve_under_root


def test_resolve_under_root_keeps_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    abs_path = (tmp_path / "abs" / "x").resolve()
    assert _resolve_under_root(root, str(abs_path)) == str(abs_path)


def test_resolve_under_root_anchors_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert _resolve_under_root(root, "downloaded_datasets") == str((root / "downloaded_datasets").resolve())


def test_find_project_root_finds_nearest_marker(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")

    found = _find_project_root(nested)
    assert found == root.resolve()
