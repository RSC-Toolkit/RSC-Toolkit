from pathlib import Path

from rsc_toolkit.datasets import _resolve_under_root


def test_resolve_under_root_keeps_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    abs_path = (tmp_path / "abs" / "x").resolve()
    assert _resolve_under_root(root, str(abs_path)) == str(abs_path)


def test_resolve_under_root_anchors_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert _resolve_under_root(root, "downloaded_datasets") == str((root / "downloaded_datasets").resolve())

