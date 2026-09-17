"""Atomic saving of best.pth / best.json / final_state.pth.

Writing the target file directly would leave a corrupted best file if the process dies mid-write (disk full, kill),
and aggregation and re-verification would lose the whole run. This uses the same tmp + os.replace convention as
last.ckpt: write to `<name>.tmp.<pid>` in the same directory, then replace, so at any moment the
target path holds either the complete previous file or the complete new file.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import torch


def _tmp_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def atomic_torch_save(obj: Any, path: os.PathLike | str) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_json_dump(obj: Any, path: os.PathLike | str, *, indent: int = 1, **kwargs: Any) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_text(json.dumps(obj, indent=indent, **kwargs))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
