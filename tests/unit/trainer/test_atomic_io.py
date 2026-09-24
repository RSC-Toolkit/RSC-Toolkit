"""Atomic save of best.pth/best.json — if the process dies mid-write the previous file stays intact and no tmp leftovers remain."""
from __future__ import annotations

import json

import pytest
import torch

from rsc_toolkit.trainer.callbacks import atomic_io


def test_atomic_writes_replace_and_leave_no_tmp(tmp_path):
    p = atomic_io.atomic_torch_save({"a": torch.ones(2)}, tmp_path / "sub" / "best.pth")
    assert torch.equal(torch.load(p, weights_only=False)["a"], torch.ones(2))
    q = atomic_io.atomic_json_dump({"epoch": 3}, tmp_path / "sub" / "best.json")
    assert json.loads(q.read_text()) == {"epoch": 3}
    assert sorted(x.name for x in (tmp_path / "sub").iterdir()) == ["best.json", "best.pth"]


def test_failed_write_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "best.pth"
    atomic_io.atomic_torch_save({"v": 1}, path)

    def boom(obj, f):
        raise OSError("disk full")

    monkeypatch.setattr(atomic_io.torch, "save", boom)
    with pytest.raises(OSError):
        atomic_io.atomic_torch_save({"v": 2}, path)
    assert torch.load(path, weights_only=False) == {"v": 1}
    assert [x.name for x in tmp_path.iterdir()] == ["best.pth"]


def test_best_checkpoint_callbacks_use_atomic_io():
    import inspect

    from rsc_toolkit.trainer.callbacks import amsd_best_checkpoint, final_state, icbhi_best_checkpoint, respirenet_vote_eval

    for mod in (icbhi_best_checkpoint, amsd_best_checkpoint, respirenet_vote_eval, final_state):
        src = inspect.getsource(mod)
        assert "torch.save(" not in src and "json.dump(" not in src, mod.__name__
