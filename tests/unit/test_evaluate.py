"""rsc_toolkit.evaluate — 체크포인트 재평가 엔트리포인트의 순수 함수들 (P2-3).

설정 복구·정제·스냅샷 선택·override·버퍼 적응·대조 판정을 고정한다.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
from torch import nn

from rsc_toolkit import evaluate as ev


def _write_snapshot(root: pathlib.Path, date: str, time: str, run_name: str, extra: str = "") -> pathlib.Path:
    d = root / "outputs" / date / time / ".hydra"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(
        "model:\n  name: ast_patchmix\n  params: {n_cls: 4}\n"
        "data:\n  name: icbhi_patchmix\n  params:\n    project_root: ${hydra:runtime.cwd}\n    data_dir: .datasets/icbhi\n"
        f"run:\n  name: {run_name}\n{extra}"
    )
    return d.parent


def test_load_snapshot_config_replaces_hydra_cwd(tmp_path):
    snap = _write_snapshot(tmp_path, "2026-09-08", "10-00-00", "r1")
    cfg = ev.load_snapshot_config(snap, tmp_path)
    assert cfg["data"]["params"]["project_root"] == str(tmp_path)
    assert cfg["run"]["name"] == "r1"


def test_find_and_pick_snapshot_by_run_name_and_ckpt_mtime(tmp_path):
    a = _write_snapshot(tmp_path, "2026-09-08", "10-00-00", "r1")
    b = _write_snapshot(tmp_path, "2026-09-08", "12-00-00", "r1")
    _write_snapshot(tmp_path, "2026-09-08", "13-00-00", "other")
    # 기본 run.name(${data.name}-${model.name}) 도 조립해서 찾는다
    c = _write_snapshot(tmp_path, "2026-09-08", "14-00-00", "${data.name}-${model.name}")
    assert ev.find_snapshot_dirs("r1", tmp_path) == [a, b]
    assert ev.find_snapshot_dirs("icbhi_patchmix-ast_patchmix", tmp_path) == [c]

    run_dir = tmp_path / "outputs" / "patchmix" / "r1"
    run_dir.mkdir(parents=True)
    best = run_dir / "best.pth"
    best.write_bytes(b"x")
    import datetime, os

    ts = datetime.datetime(2026, 9, 8, 11, 30).timestamp()  # a 이후, b 이전에 저장된 best.pth → a
    os.utime(best, (ts, ts))
    assert ev.pick_snapshot([a, b], best) == a
    ts = datetime.datetime(2026, 9, 8, 12, 30).timestamp()
    os.utime(best, (ts, ts))
    assert ev.pick_snapshot([a, b], best) == b
    assert ev.pick_snapshot([], best) is None

    cfg, source = ev.resolve_run_config(run_dir, tmp_path)
    assert cfg["run"]["name"] == "r1" and source == f"hydra:{b}"
    cfg, source = ev.resolve_run_config(run_dir, tmp_path, hydra_dir=a)
    assert source == f"hydra:{a}"


def test_load_log_config_parses_last_resolved_block(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "2026-09-08 10:00:00 [INFO] start\nmodel:\n  name: old\ndata:\n  name: x\n[stuff]\n"
        "model:\n  name: ast_patchmix\n  params:\n    n_cls: 4\ndata:\n  name: icbhi_patchmix\nseed: 1\n"
        "2026-09-08 10:00:01 [INFO] training\n"
    )
    cfg = ev.load_log_config(log)
    assert cfg["model"]["name"] == "ast_patchmix" and cfg["seed"] == 1
    (tmp_path / "empty.log").write_text("nothing here\n")
    assert ev.load_log_config(tmp_path / "empty.log") is None


def test_sanitize_config_drops_writing_callbacks_and_absolutizes(tmp_path):
    cfg = {
        "resume": "auto",
        "seed": 1,
        "data": {"params": {"data_dir": ".datasets/icbhi", "features": {"cache_dir": "downloaded_datasets/features_cache"}}},
        "model": {"params": {"encoder_ckpt": "outputs/x/best.pth", "name": "not/a/path"}},
        "trainer": {"params": {"max_epochs": 5, "callbacks": [
            {"_target_": "rsc_toolkit.trainer.callbacks.icbhi_best_checkpoint.ICBHIBestScoreCheckpoint", "dirpath": "outputs/x"},
            {"_target_": "rsc_toolkit.trainer.callbacks.respirenet_vote_eval.RespireNetVoteEval", "dirpath": "outputs/x"},
            {"_target_": "rsc_toolkit.trainer.callbacks.ema.EMA"},
        ]}},
    }
    out, dropped = ev.sanitize_config(cfg, tmp_path / "sb", "r1", tmp_path)
    assert "resume" not in out and cfg.get("resume") == "auto"  # 원본 비파괴
    assert out["data"]["params"]["data_dir"] == str(tmp_path / ".datasets/icbhi")
    assert out["data"]["params"]["features"]["cache_dir"] == str(tmp_path / "downloaded_datasets/features_cache")
    assert out["model"]["params"]["encoder_ckpt"] == str(tmp_path / "outputs/x/best.pth")
    assert out["model"]["params"]["name"] == "not/a/path"
    tp = out["trainer"]["params"]
    assert dropped == ["ICBHIBestScoreCheckpoint", "EMA"]
    assert [c["_target_"].rsplit(".", 1)[-1] for c in tp["callbacks"]] == ["RespireNetVoteEval"]
    assert tp["callbacks"][0]["dirpath"] == str(tmp_path / "sb" / "callback_out" / "r1")
    assert tp["logger"] is False and tp["enable_checkpointing"] is False and tp["max_epochs"] == 5
    assert tp["default_root_dir"] == str(tmp_path / "sb" / "lightning")


def test_apply_overrides_dotlist():
    cfg = {"data": {"params": {"batch_size": 8, "shuffle_train": True}}}
    out = ev.apply_overrides(cfg, ["data.params.batch_size=16", "data.params.shuffle_train=false"])
    assert out["data"]["params"] == {"batch_size": 16, "shuffle_train": False}
    assert ev.apply_overrides(cfg, []) is cfg


def test_adapt_buffer_shapes_and_load_checkpoint(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("all_predictions", torch.zeros(3))   # AMS-D per-sample 버퍼 (학습 샘플 수에 따라 다름)

    m = M()
    sd = {"lin.weight": torch.ones(2, 2), "lin.bias": torch.zeros(2), "all_predictions": torch.arange(5.0)}
    torch.save({"model": sd}, tmp_path / "best.pth")
    info = ev.load_checkpoint(m, tmp_path / "best.pth")
    assert info["load"] == "strict_ok" and info["buffer_adapted"] == ["all_predictions: (3,) -> (5,)"]
    assert torch.equal(m.all_predictions, torch.arange(5.0))


def test_adapt_buffer_shapes_leaves_other_buffers_alone(tmp_path):
    """per-sample 버퍼가 아닌 모양 불일치(BN running stat 등)는 모델 구성이 다른 것 — 0 으로 덮지 말고 strict 실패."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("running_mean", torch.zeros(3))

    m = M()
    sd = {"lin.weight": torch.ones(2, 2), "lin.bias": torch.zeros(2), "running_mean": torch.arange(5.0)}
    assert ev.adapt_buffer_shapes(m, sd) == []
    torch.save({"model": sd}, tmp_path / "best.pth")
    info = ev.load_checkpoint(m, tmp_path / "best.pth")
    assert info["load"] != "strict_ok"

    torch.save({"lin.weight": torch.ones(2, 2)}, tmp_path / "partial.pth")  # state_dict 자체 + 키 누락
    info = ev.load_checkpoint(M(), tmp_path / "partial.pth")
    assert info["load"] == "strict_fail" and "lin.bias" in info["load_missing"]


def test_compare_standard_and_vote_keys():
    status, diffs, worst = ev.compare({"S_p": 70.0, "S_e": 40.0, "icbhi_score": 55.0}, {"S_p": 70.01, "S_e": 40.0, "icbhi_score": 55.0}, 0.05)
    assert status == "PASS" and worst == pytest.approx(0.01)
    status, _, _ = ev.compare({"S_p": 70.0, "S_e": 40.0, "icbhi_score": 55.0}, {"S_p": 71.0, "S_e": 40.0, "icbhi_score": 55.0}, 0.05)
    assert status == "MISMATCH"
    status, diffs, _ = ev.compare({"cycle_acc": 58.0, "icbhi_score": 56.0}, {"vote/cycle_acc": 58.0, "vote/icbhi_score": 56.0}, 0.05)
    assert status == "PASS" and set(diffs) == {"cycle_acc", "icbhi_score"}
    status, diffs, worst = ev.compare({"S_p": 70.0}, {}, 0.05)
    assert status == "ERROR" and worst is None and diffs["missing_eval_keys"] == ["S_p"]


def test_make_sandbox_symlinks_read_only_roots_only(tmp_path):
    (tmp_path / "rsc_toolkit").mkdir()
    (tmp_path / "outputs").mkdir()
    sb = tmp_path / "sb"
    ev.make_sandbox(sb, tmp_path)
    assert (sb / "rsc_toolkit").is_symlink() and not (sb / "outputs").exists()
    assert (sb / "lightning").is_dir() and (sb / "callback_out").is_dir()
    ev.make_sandbox(sb, tmp_path)  # 멱등


def test_readonly_chain_guard_records_instead_of_writing(tmp_path):
    from rsc_toolkit.datasets.feature_cache import FeatureCache

    cache = FeatureCache(cache_dir=tmp_path)
    cache.configure({"n_mels": 64}, processor_name="Mel", audio_chain={"dataset_name": "d", "sampling_rate": 16000})
    ledger_path = cache.features_dir / "_chain.json"
    before = ledger_path.read_text()
    with ev.readonly_chain_guard() as notes:
        cache.configure({"n_mels": 64}, processor_name="Mel", audio_chain={"dataset_name": "d", "sampling_rate": 8000})  # 어긋남
        cache.configure({"n_mels": 32}, processor_name="Mel", audio_chain={"dataset_name": "d", "sampling_rate": 8000})  # 대장 없음
    assert len(notes) == 1 and notes[0]["dataset"] == "d" and notes[0]["requested"] == {"sampling_rate": 8000}
    assert ledger_path.read_text() == before
    assert not (cache.features_dir / "_chain.json").exists()  # 새 키 디렉터리에 대장을 쓰지 않았다
    with pytest.raises(RuntimeError):  # 가드 해제 후엔 원래대로 예외
        FeatureCache(cache_dir=tmp_path).configure({"n_mels": 64}, processor_name="Mel", audio_chain={"dataset_name": "d", "sampling_rate": 8000})


def test_evaluate_run_reports_missing_ckpt_and_config(tmp_path):
    run_dir = tmp_path / "outputs" / "g" / "r9"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ev.evaluate_run(run_dir, project_root=tmp_path)
    (run_dir / "best.pth").write_bytes(b"x")
    rec = ev.evaluate_run(run_dir, project_root=tmp_path)
    assert rec["status"] == "SKIP" and rec["config_source"] == "none"


def test_cli_parses_and_reports_skip(tmp_path, capsys):
    run_dir = tmp_path / "outputs" / "g" / "r9"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pth").write_bytes(b"x")
    rc = ev.main([str(run_dir), "--project-root", str(tmp_path), "--out", str(tmp_path / "eval.json"), "--quiet"])
    assert rc == 1
    assert json.loads((tmp_path / "eval.json").read_text())["status"] == "SKIP"
    assert "SKIP" in capsys.readouterr().out


def test_evaluate_run_with_overrides_reports_diffs_but_status_ok(tmp_path, monkeypatch):
    """override 가 있으면 판정(PASS/MISMATCH) 대신 OK 이되 best.json 대조값은 돌려준다 (튜토리얼의 공개 weight 재평가가 쓴다)."""
    import json
    from rsc_toolkit import evaluate as ev
    run_dir = tmp_path / "outputs" / "g" / "r"; run_dir.mkdir(parents=True)
    (run_dir / "best.pth").write_bytes(b"x")
    (run_dir / "best.json").write_text(json.dumps({"epoch": 1, "S_p": 80.0, "S_e": 40.0, "icbhi_score": 60.0}))
    (run_dir / ".hydra").mkdir(); (run_dir / ".hydra" / "config.yaml").write_text("data: {params: {data_dir: a}}\ntrainer: {params: {}}\n")
    monkeypatch.setattr(ev, "build_and_validate", lambda cfg, ckpt, sandbox, name: {"load": "strict_ok", "metrics": {"S_p": 80.0, "S_e": 40.0, "icbhi_score": 60.0}})
    monkeypatch.setattr(ev, "install_readonly_chain_guard", lambda notes: (lambda: None))   # restore 콜백을 돌려줘야 한다
    rec = ev.evaluate_run(run_dir, project_root=tmp_path, sandbox=tmp_path / "sb", overrides=["data.params.data_dir=b"])
    assert rec["status"] == "OK" and rec["compared_with_overrides"] is True and rec["max_abs_diff"] == 0.0 and "S_e" in rec["diffs"]
    rec2 = ev.evaluate_run(run_dir, project_root=tmp_path, sandbox=tmp_path / "sb2")
    assert rec2["status"] == "PASS" and rec2["compared_with_overrides"] is False
