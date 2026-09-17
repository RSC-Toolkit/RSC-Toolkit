"""Single source of path resolution — so the toolkit also works when installed outside the repository.

Two things are decided here:

1. **Pretrained checkpoints** (`pretrained_models/upstream/…`). Each model has a list of candidate paths relative to cwd and
   downloads the file if none exists; the `RSCTK_PRETRAINED_ROOT` environment variable is checked first, so several checkouts
   can share one weights directory and users of the installed package can keep weights anywhere. Resolution order:
     explicit path (pretrained_path in the config)    → as is (relative to cwd; if missing, also looked up under ROOT)
     for each candidate path c                        → ROOT/<c without the 'pretrained_models/' prefix> → ROOT/<basename> → cwd/c
   The download destination follows the same rule (under ROOT if set).

2. **Project root**. `Path(__file__).parents[2]` points to site-packages in an installed package, so it cannot be used to
   locate the repository root. `project_root_from()` resolves: argument → `RSCTK_PROJECT_ROOT` → cwd (if it has repository markers) → package parent (development checkout) → cwd.
"""
from __future__ import annotations

import os
import pathlib
from typing import Iterable, Optional, Sequence, Union

PRETRAINED_ROOT_ENV = "RSCTK_PRETRAINED_ROOT"
PROJECT_ROOT_ENV = "RSCTK_PROJECT_ROOT"
_PRETRAINED_PREFIX = "pretrained_models/"
_ROOT_MARKERS = ("pyproject.toml", ".git", "rsc_toolkit", ".datasets", "pretrained_models")

PathLike = Union[str, os.PathLike]


def pretrained_root() -> Optional[pathlib.Path]:
    """`RSCTK_PRETRAINED_ROOT` (None if unset)."""
    raw = os.environ.get(PRETRAINED_ROOT_ENV, "").strip()
    return pathlib.Path(raw).expanduser() if raw else None


def _under_root(root: pathlib.Path, candidate: str) -> Iterable[pathlib.Path]:
    rel = candidate[len(_PRETRAINED_PREFIX):] if candidate.startswith(_PRETRAINED_PREFIX) else candidate
    yield root / rel
    yield root / pathlib.Path(candidate).name


def resolve_pretrained(candidates: Sequence[str] = (), explicit: Optional[PathLike] = None) -> Optional[pathlib.Path]:
    """Return an existing checkpoint path (None if not found — the caller downloads it or raises).

    `explicit` takes precedence: an absolute or existing relative path is returned as is; a missing relative path is also
    looked up under ROOT, and if still not found **that path is returned unchanged** (so the caller's error message names it).
    """
    root = pretrained_root()
    if explicit is not None:
        p = pathlib.Path(explicit).expanduser()
        if p.exists() or p.is_absolute():
            return p
        if root is not None:
            for q in _under_root(root, str(p)):
                if q.exists():
                    return q
        return p
    for cand in candidates:
        if root is not None:
            for q in _under_root(root, cand):
                if q.exists():
                    return q
        p = pathlib.Path(cand).expanduser()
        if p.exists():
            return p
    return None


def download_target(candidate: str) -> pathlib.Path:
    """Download destination for a checkpoint — under ROOT if set (prefix removed), otherwise the candidate path relative to cwd."""
    root = pretrained_root()
    if root is not None:
        rel = candidate[len(_PRETRAINED_PREFIX):] if candidate.startswith(_PRETRAINED_PREFIX) else candidate
        return root / rel
    return pathlib.Path(candidate)


def project_root_from(project_root: Optional[PathLike] = None) -> pathlib.Path:
    """Repository/work root: argument → RSCTK_PROJECT_ROOT → cwd (if it has marker files) → package parent (development checkout) → cwd."""
    if project_root:
        return pathlib.Path(project_root).expanduser().resolve()
    env = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if env:
        return pathlib.Path(env).expanduser().resolve()
    cwd = pathlib.Path.cwd().resolve()
    if any((cwd / m).exists() for m in _ROOT_MARKERS):
        return cwd
    dev = pathlib.Path(__file__).resolve().parents[1]
    if (dev / "pyproject.toml").exists():
        return dev
    return cwd


__all__ = ["PRETRAINED_ROOT_ENV", "PROJECT_ROOT_ENV", "pretrained_root", "resolve_pretrained", "download_target", "project_root_from"]
