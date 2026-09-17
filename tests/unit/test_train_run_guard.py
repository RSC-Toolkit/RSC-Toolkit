"""Guard against re-launching the same run.name — existing best.pth/best.json are not silently overwritten."""
from __future__ import annotations

import pytest

from rsc_toolkit.train import existing_run_artifacts, guard_existing_run
from rsc_toolkit.trainer.callbacks.icbhi_best_checkpoint import ICBHIBestScoreCheckpoint
from rsc_toolkit.trainer.callbacks.resume_checkpoint import EpochResumeCheckpoint


class _Trainer:
    def __init__(self, callbacks):
        self.callbacks = callbacks


def test_nothing_found_on_fresh_run(tmp_path):
    t = _Trainer([ICBHIBestScoreCheckpoint(dirpath=str(tmp_path / "run"))])
    assert existing_run_artifacts(t) == []
    guard_existing_run(t, {"run": {"name": "x"}}, resuming=False)     # no exception


def test_existing_best_blocks_unless_overwrite_or_resume(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "best.pth").write_bytes(b"x"); (d / "best.json").write_text("{}")
    (d / "last.ckpt").write_bytes(b"y")
    t = _Trainer([ICBHIBestScoreCheckpoint(dirpath=str(d)), EpochResumeCheckpoint(dirpath=str(d))])
    assert [p.name for p in existing_run_artifacts(t)] == ["best.json", "best.pth"]      # last.ckpt is excluded
    with pytest.raises(RuntimeError, match="run.overwrite=true"):
        guard_existing_run(t, {"run": {"name": "x"}}, resuming=False)
    guard_existing_run(t, {"run": {"name": "x"}}, resuming=True)                   # resuming is allowed
    guard_existing_run(t, {"run": {"name": "x", "overwrite": True}}, resuming=False)   # explicitly allowed


def test_final_state_and_metrics_path_callbacks_are_seen(tmp_path):
    from rsc_toolkit.trainer.callbacks.final_state import FinalStateCheckpoint
    from rsc_toolkit.trainer.callbacks.gapaug_final_metrics import FinalMetricsToBestJson

    d = tmp_path / "run"; d.mkdir()
    (d / "final_state.pth").write_bytes(b"x"); (d / "best.json").write_text("{}")
    cbs = [FinalStateCheckpoint(dirpath=str(d)), FinalMetricsToBestJson(dirpath=str(d))]
    assert [p.name for p in existing_run_artifacts(_Trainer(cbs))] == ["best.json", "final_state.pth"]
