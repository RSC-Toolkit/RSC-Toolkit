# -*- coding: utf-8 -*-
"""
논문에 쓰이는 weight 파일(outputs/<group>/<run>/best.pth) 전수 검증.

각 run 에 대해
  1. 그 run 의 설정 스냅샷(outputs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml, 없으면 run 로그에 찍힌 resolved config)
     으로 모델·손실·지표·datamodule 을 **학습 때와 같은 팩토리**로 재구성하고,
  2. best.pth 의 state_dict 를 strict=True 로 로드한 뒤,
  3. 공식 test 스플릿을 trainer.validate 로 재평가해 best.json 의 S_p/S_e/Score 와 대조한다.

outputs/ 는 **읽기만 한다**: 프로세스의 CWD 를 샌드박스로 옮기고(입력 디렉터리만 심링크),
체크포인트·json 을 쓰는 콜백은 제거하거나 샌드박스로 리다이렉트하며, 로거는 끈다.

  .venv/bin/python scripts/verify_paper_weights.py --out DIR [--group G ...] [--dataset icbhi|sprsound]
                                                   [--run NAME ...] [--shard i/n] [--smoke] [--tol 0.05]
  .venv/bin/python scripts/verify_paper_weights.py --out DIR --summarize     # 샤드 결과 병합 → results.csv/summary.md

판정: PASS(|Δ| ≤ tol) · MISMATCH · LOAD_FAIL(strict 로드 실패) · ERROR · SKIP(설정 복구 불가).
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 입력으로만 쓰는 최상위 디렉터리 — 샌드박스에 심링크로 노출한다. outputs/ 는 일부러 뺀다.
READ_ONLY_ROOTS = ("rsc_toolkit", ".datasets", "downloaded_datasets", "pretrained_models", "sample_data")
# config 안의 상대 경로를 절대 경로로 바꿀 접두사 (outputs/ 는 encoder_ckpt 같은 **입력** 에만 등장한다 —
# 콜백 dirpath 는 아래에서 따로 처리한다).
ABS_PREFIXES = tuple(f"{r}/" for r in READ_ONLY_ROOTS) + ("outputs/",)
# validate 에서 살려 두는 콜백 — 지표·손실 재현에 필요하거나 무해한 것만. 나머지(체크포인트·json 저장, MA/EMA,
# LR 스케줄, 샘플러 epoch 등)는 전부 버린다.
KEEP_CALLBACKS = {"AMSDEpochCallback", "DomainTransferStages", "ProtoNetClassWeight", "RespireNetVoteEval"}
STD_KEYS = ("S_p", "S_e", "icbhi_score")
PAPER_KEYS = ("paper_S_p", "paper_S_e", "paper_score", "paper_acc")
VOTE_MAP = {"S_p": "vote/S_p", "S_e": "vote/S_e", "icbhi_score": "vote/icbhi_score", "cycle_acc": "vote/cycle_acc"}

log = logging.getLogger("verify")


# ----------------------------------------------------------------------------- 설정 복구
def load_snapshot_config(hydra_dir: pathlib.Path) -> dict:
    import yaml
    from omegaconf import OmegaConf

    text = (hydra_dir / ".hydra" / "config.yaml").read_text()
    text = text.replace("${hydra:runtime.cwd}", str(ROOT))
    cfg = OmegaConf.create(yaml.safe_load(text))
    return OmegaConf.to_container(cfg, resolve=True)


def find_run_log(run_name: str) -> pathlib.Path | None:
    hits = sorted((ROOT / "outputs").glob(f"*/logs/{run_name}.log"))
    return hits[-1] if hits else None


def load_log_config(log_path: pathlib.Path) -> dict | None:
    """train.py 가 시작 직후 찍는 `OmegaConf.to_yaml(cfg, resolve=True)` 블록을 파싱한다 (마지막 블록)."""
    import yaml
    from omegaconf import OmegaConf

    lines = log_path.read_text(errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln == "model:"]
    if not starts:
        return None
    block = []
    for ln in lines[starts[-1]:]:
        # 최상위 키(소문자 key:) 또는 들여쓰기 줄만 YAML 블록에 속한다. 로그 줄·경고·Lightning 출력에서 멈춘다.
        if ln.startswith((" ", "-")) or re.match(r"^[a-z_]+:( .*)?$", ln):
            block.append(ln)
        else:
            break
    cfg = yaml.safe_load("\n".join(block))
    if not isinstance(cfg, dict) or "model" not in cfg or "data" not in cfg:
        return None
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def _absolutize(obj, skip_path=()):
    """상대 경로 문자열을 ROOT 기준 절대 경로로. 콜백(dirpath) 서브트리는 건너뛴다."""
    if isinstance(obj, dict):
        return {k: (v if k in skip_path else _absolutize(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_absolutize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(ABS_PREFIXES):
        return str(ROOT / obj)
    return obj


def sanitize_config(cfg: dict, sandbox: pathlib.Path, run_name: str) -> tuple[dict, list[str]]:
    cfg = copy.deepcopy(cfg)
    cfg.pop("resume", None)
    tp = cfg.setdefault("trainer", {}).setdefault("params", {})
    callbacks = tp.get("callbacks") or []
    tp_wo_cb = {k: v for k, v in tp.items() if k != "callbacks"}
    cfg["trainer"]["params"] = tp_wo_cb
    cfg = _absolutize(cfg)
    tp = cfg["trainer"]["params"]

    kept, dropped = [], []
    for cb in callbacks:
        name = str(cb.get("_target_", "")).rsplit(".", 1)[-1]
        if name in KEEP_CALLBACKS:
            cb = dict(cb)
            if "dirpath" in cb:  # RespireNetVoteEval — best.pth/best.json 을 쓰므로 샌드박스로
                cb["dirpath"] = str(sandbox / "callback_out" / run_name)
            kept.append(cb)
        else:
            dropped.append(name)
    tp["callbacks"] = kept
    tp["logger"] = False
    tp["enable_checkpointing"] = False
    tp["enable_progress_bar"] = False
    tp["enable_model_summary"] = False
    tp["default_root_dir"] = str(sandbox / "lightning")
    return cfg, dropped


# ----------------------------------------------------------------------------- 평가
def _adapt_buffer_shapes(model, sd: dict) -> list[str]:
    """ckpt 의 버퍼 모양이 새 모델과 다르면(AMSD 의 per-sample 버퍼 등) 버퍼만 ckpt 모양으로 바꾼다."""
    changed = []
    buffers = dict(model.named_buffers())
    for name, tensor in sd.items():
        cur = buffers.get(name)
        if cur is None or tuple(cur.shape) == tuple(tensor.shape):
            continue
        mod_name, _, attr = name.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model
        mod._buffers[attr] = torch.zeros_like(tensor, device=cur.device)
        changed.append(f"{name}: {tuple(cur.shape)} -> {tuple(tensor.shape)}")
    return changed


CHAIN_NOTES: list = []


def _install_readonly_chain_guard() -> None:
    """FeatureCache._check_audio_chain 을 읽기 전용으로 바꾼다.

    원본은 대장(_chain.json)이 없으면 **실제 캐시 디렉터리에 새로 쓰고**, 어긋나면 예외를 던진다.
    검증은 캐시를 읽기만 해야 하므로 쓰기를 없애고, 어긋남은 예외 대신 기록으로 남긴다
    (waveform 종류는 캐시를 아예 안 쓰므로 어긋나도 평가에 영향이 없다 — 그 사실도 기록에 담는다)."""
    import json as _json

    from rsc_toolkit.datasets import feature_cache as fc_mod

    def _readonly_check(self, chain):
        dataset = str(chain.pop("dataset_name", "?"))
        path = self.features_dir / "_chain.json"
        if not path.exists():
            return
        try:
            known = (_json.loads(path.read_text()) or {}).get(dataset)
        except Exception:
            return
        if known is not None and known != chain:
            n_files = sum(1 for p in self.features_dir.iterdir() if p.name != "_chain.json")
            CHAIN_NOTES.append({"cache": str(self.features_dir), "dataset": dataset, "ledger": known,
                                "requested": chain, "cached_files": n_files})

    fc_mod.FeatureCache._check_audio_chain = _readonly_check


def build_and_validate(cfg: dict, ckpt_path: pathlib.Path, sandbox: pathlib.Path, run_name: str) -> dict:
    import lightning as L

    from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
    from rsc_toolkit.trainer.factories.build_loss import build_loss
    from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
    from rsc_toolkit.trainer.factories.build_model import build_model
    from rsc_toolkit.trainer.factories.build_trainer import build_trainer

    info: dict = {}
    seed = cfg.get("seed")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)

    model = build_model(copy.deepcopy(cfg["model"]))
    loss_func = build_loss(copy.deepcopy(cfg["loss"]))
    train_fns, val_fns = build_train_val_metric_factories(copy.deepcopy(cfg.get("metrics", {})))
    datamodule = build_datamodule(copy.deepcopy(cfg["data"]))
    trainer = build_trainer(copy.deepcopy(cfg["trainer"]))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    info["buffer_adapted"] = _adapt_buffer_shapes(model, sd)
    try:
        model.load_state_dict(sd, strict=True)
        info["load"] = "strict_ok"
    except RuntimeError as exc:
        res = model.load_state_dict(sd, strict=False)
        info["load"] = "strict_fail"
        info["load_missing"] = list(res.missing_keys)[:20]
        info["load_unexpected"] = list(res.unexpected_keys)[:20]
        info["load_error"] = str(exc)[:500]
        return info

    adapter_cfg = cfg.get("adapter") or {}
    if adapter_cfg.get("name") == "split_iter":
        from rsc_toolkit.trainer.adapters.split_iter_adapter import SplitIterAdapter

        pl_module = SplitIterAdapter(rsc_model=model, loss_func=loss_func, train_metric_fns=train_fns,
                                     val_metric_fns=val_fns, optimizer_cfg=cfg["optimizer"],
                                     **(adapter_cfg.get("params") or {}))
    else:
        pl_module = PLAdapter(rsc_model=model, loss_func=loss_func, train_metric_fns=train_fns,
                              val_metric_fns=val_fns, optimizer_cfg=cfg["optimizer"])

    # Lightning 은 validate 에 setup("validate") 를 부르는데 툴킷 datamodule 들은 fit/test 만 안다 → None(전부) 로 매핑.
    orig_setup = datamodule.setup
    datamodule.setup = lambda stage=None: orig_setup(None if stage == "validate" else stage)  # type: ignore[assignment]

    out = trainer.validate(pl_module, datamodule=datamodule, verbose=False)
    metrics = {}
    for d in out or []:
        metrics.update({k: float(v) for k, v in d.items()})
    metrics.update({k: float(v) for k, v in trainer.callback_metrics.items()})
    info["metrics"] = metrics
    return info


def compare(expected: dict, metrics: dict, tol: float) -> tuple[str, dict, float | None]:
    if "cycle_acc" in expected:  # RespireNet — 사이클 투표 지표
        pairs = {k: (expected.get(k), metrics.get(v)) for k, v in VOTE_MAP.items()}
    else:
        keys = list(STD_KEYS) + [k for k in PAPER_KEYS if k in expected]
        pairs = {k: (expected.get(k), metrics.get(k)) for k in keys}
    diffs, missing, worst = {}, [], None
    for k, (e, m) in pairs.items():
        if e is None:
            continue
        if m is None:
            missing.append(k)
            continue
        d = abs(float(e) - float(m))
        diffs[k] = {"expected": float(e), "evaluated": float(m), "diff": d}
        worst = d if worst is None else max(worst, d)
    if not diffs:
        return "ERROR", {"missing_eval_keys": missing}, None
    status = "PASS" if worst <= tol else "MISMATCH"
    if missing:
        diffs["missing_eval_keys"] = missing
    return status, diffs, worst


# ----------------------------------------------------------------------------- MVST 융합 (provenance=self)
def verify_fusion(row: dict, run_dir: pathlib.Path, tol: float, device: str) -> dict:
    """mvst_fusion.py 산출물: features.pt(뷰 특징) + best.pth(FuseMLP). 저장된 융합 가중치를 test 특징에 적용해 대조."""
    from rsc_toolkit.methods.mvst_fusion import FuseMLP, get_score

    best = json.loads((run_dir / "best.json").read_text())
    feats_path = run_dir / "features.pt"
    if not feats_path.exists():
        return {"status": "SKIP", "note": "features.pt 없음 — 뷰 ckpt 로 특징을 다시 뽑아야 한다"}
    ckpts = best.get("ckpts") or {}
    missing_views = [v for v, p in ckpts.items() if not (ROOT / p).exists()]
    packed = torch.load(feats_path, map_location="cpu", weights_only=False)
    te_feats, te_labels = packed["test"], packed["test_labels"]
    n_cls = int(best.get("n_cls") or (int(te_labels.max()) + 1))
    model = FuseMLP(num_views=len(te_feats), num_classes=n_cls)
    sd = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    use_amp = str(device).startswith("cuda")
    hits, counts = [0.0] * n_cls, [0.0] * n_cls
    ds = torch.utils.data.TensorDataset(*te_feats, te_labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for *fs, y in loader:
            fs = [f.to(device) for f in fs]
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(*fs).argmax(1)
            for p, t in zip(pred.tolist(), y.tolist()):
                counts[t] += 1
                if p == t:
                    hits[t] += 1
    sp, se, sc = get_score(hits, counts)
    status, diffs, worst = compare(best, {"S_p": sp, "S_e": se, "icbhi_score": sc}, tol)
    out = {"status": status, "diffs": diffs, "max_abs_diff": worst, "load": "strict_ok"}
    if missing_views:
        out["note"] = f"참조 뷰 ckpt 없음: {missing_views}"
    return out


# ----------------------------------------------------------------------------- 드라이버
def select_rows(args) -> list[dict]:
    rows = list(csv.DictReader(open(args.master)))
    if args.group:
        rows = [r for r in rows if r["paper_group"] in set(args.group)]
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset or (not r["dataset"] and args.dataset in r["run_name"])]
    if args.run:
        rows = [r for r in rows if r["run_name"] in set(args.run)]
    if args.smoke:
        seen, picked = set(), []
        for r in rows:
            key = (r["paper_group"], r["provenance"])
            if key not in seen:
                seen.add(key)
                picked.append(r)
        rows = picked
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows = rows[i::n]
    if args.limit:
        rows = rows[: args.limit]
    return rows


VARIANT_SUFFIXES = ("-unweighted", "-oldrecipe", "-m2dproto-discarded")


def resolve_config(row: dict, master_rows: list[dict] | None = None) -> tuple[dict | None, str]:
    if row["provenance"] == "hydra" and row.get("hydra_dir"):
        return load_snapshot_config(ROOT / row["hydra_dir"]), "hydra"
    log_path = find_run_log(row["run_name"])
    if log_path is not None:
        cfg = load_log_config(log_path)
        if cfg is not None:
            return cfg, f"log:{log_path.relative_to(ROOT)}"
    # 보존용 변형 run(*-unweighted 등)은 결과 디렉터리만 개명돼 스냅샷·로그가 없다. 지표(S_p/S_e/Score)는
    # 손실 함수와 무관하므로 접미사를 뗀 형제 run 의 설정으로 재구성해 weight 자체를 검증한다 —
    # 재현되지 않으면 데이터 파이프라인이 달랐다는 뜻이라 그것도 정보다.
    for suf in VARIANT_SUFFIXES:
        if row["run_name"].endswith(suf) and master_rows is not None:
            sibling = row["run_name"][: -len(suf)]
            # 같은 이름의 형제가 없으면(예: 가중 재실행이 완주하지 않은 seed) 다른 seed 의 형제로 —
            # seed·run.name 외의 설정은 시리즈 안에서 같다. 어느 형제를 썼는지 출처에 남긴다.
            candidates = [sibling] + [re.sub(r"-s\d+$", f"-s{k}", sibling) for k in range(1, 6)]
            for cand in candidates:
                for r in master_rows:
                    if r["run_name"] == cand and r["provenance"] == "hydra" and r.get("hydra_dir"):
                        return load_snapshot_config(ROOT / r["hydra_dir"]), f"sibling:{cand}"
    return None, "none"


def make_sandbox(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in READ_ONLY_ROOTS:
        src, dst = ROOT / name, path / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    (path / "lightning").mkdir(exist_ok=True)
    (path / "callback_out").mkdir(exist_ok=True)


def _visible_gpu_index() -> str | None:
    """CUDA_VISIBLE_DEVICES 의 첫 항목(물리 index). nvidia-smi -i 에 쓴다."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = raw.split(",")[0].strip()
    return first if first.isdigit() else None


def wait_for_cool_gpu(max_temp: int, resume_temp: int) -> float:
    """run 사이의 온도 게이트 — GPU 가 max_temp 이상이면 resume_temp 이하로 식을 때까지 기다린다.

    평가 루프는 학습보다 GPU 를 쉬지 않고 100% 쓰기 때문에 동시 4개에서도 5~10분이면 80°C 에 닿는다
    (서버 규칙: 80°C 초과 금지). 워커를 죽였다 살리는 대신 스스로 쉰다."""
    idx = _visible_gpu_index()
    if max_temp <= 0 or idx is None:
        return 0.0
    waited, gate_open = 0.0, True
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            temp = int(out.splitlines()[0])
        except Exception:
            return waited
        if gate_open and temp < max_temp:
            return waited
        if not gate_open and temp <= resume_temp:
            return waited
        gate_open = False
        time.sleep(15)
        waited += 15


def run_one(row: dict, args, sandbox: pathlib.Path) -> dict:
    t0 = time.time()
    run_dir = ROOT / row["run_dir"]
    rec = {"group": row["paper_group"], "run": row["run_name"], "dataset": row["dataset"],
           "provenance": row["provenance"], "run_dir": row["run_dir"], "status": "ERROR"}
    try:
        best_json = json.loads((run_dir / "best.json").read_text())
        rec["expected"] = {k: best_json[k] for k in STD_KEYS + PAPER_KEYS + ("cycle_acc", "epoch") if k in best_json}
        if row["provenance"] == "self":
            rec.update(verify_fusion(row, run_dir, args.tol, args.device))
            rec["config_source"] = "self"
            return rec
        cfg, source = resolve_config(row, args.master_rows)
        rec["config_source"] = source
        if cfg is None:
            rec["status"] = "SKIP"
            rec["note"] = "설정 스냅샷도 run 로그도 없다 — 재구성 불가"
            return rec
        cfg, dropped = sanitize_config(cfg, sandbox, row["run_name"])
        rec["dropped_callbacks"] = dropped
        CHAIN_NOTES.clear()
        info = build_and_validate(cfg, run_dir / "best.pth", sandbox, row["run_name"])
        if CHAIN_NOTES:
            rec["cache_chain_mismatch"] = list(CHAIN_NOTES)
        rec.update({k: v for k, v in info.items() if k != "metrics"})
        if info.get("load") != "strict_ok":
            rec["status"] = "LOAD_FAIL"
            return rec
        status, diffs, worst = compare(best_json, info["metrics"], args.tol)
        rec["status"], rec["diffs"], rec["max_abs_diff"] = status, diffs, worst
        return rec
    except Exception as exc:  # 한 run 의 실패가 전체를 멈추지 않게
        rec["status"] = "ERROR"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:800]
        rec["traceback"] = traceback.format_exc()[-2000:]
        return rec
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
        rec["ts"] = time.time()  # 같은 run 을 다시 돌리면 최신 기록이 이긴다 (summarize)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(out_dir: pathlib.Path) -> None:
    recs = []
    for p in sorted(out_dir.glob("results-*.jsonl")):
        recs += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    by_run: dict = {}
    for r in recs:  # 같은 run 의 기록이 여럿이면(단건 재실행) 가장 최근 ts 가 이긴다
        if r["run"] not in by_run or r.get("ts", 0) >= by_run[r["run"]].get("ts", 0):
            by_run[r["run"]] = r
    recs = sorted(by_run.values(), key=lambda r: (r["group"], r["run"]))
    cols = ["group", "run", "dataset", "provenance", "status", "max_abs_diff", "exp_S_p", "exp_S_e", "exp_score",
            "eval_S_p", "eval_S_e", "eval_score", "config_source", "load", "seconds", "note"]
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            d = r.get("diffs") or {}
            g = lambda k, f: (d.get(k) or {}).get(f, "")  # noqa: E731
            w.writerow([r["group"], r["run"], r.get("dataset", ""), r.get("provenance", ""), r["status"],
                        "" if r.get("max_abs_diff") is None else f"{r['max_abs_diff']:.4f}",
                        g("S_p", "expected"), g("S_e", "expected"), g("icbhi_score", "expected"),
                        g("S_p", "evaluated"), g("S_e", "evaluated"), g("icbhi_score", "evaluated"),
                        r.get("config_source", ""), r.get("load", ""), r.get("seconds", ""),
                        (r.get("note") or r.get("error") or "")
                        + ("" if not r.get("cache_chain_mismatch") else " [cache-chain-mismatch: "
                           + "; ".join(f"{m['cached_files']} files" for m in r["cache_chain_mismatch"]) + "]")])
    counts = collections.Counter(r["status"] for r in recs)
    by_group = collections.defaultdict(collections.Counter)
    for r in recs:
        by_group[r["group"]][r["status"]] += 1
    lines = [f"# weight 검증 요약 ({len(recs)} runs)", "",
             "전체: " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "| group | PASS | MISMATCH | LOAD_FAIL | ERROR | SKIP |", "|---|---|---|---|---|---|"]
    for g, c in sorted(by_group.items()):
        lines.append(f"| {g} | {c['PASS']} | {c['MISMATCH']} | {c['LOAD_FAIL']} | {c['ERROR']} | {c['SKIP']} |")
    bad = [r for r in recs if r["status"] != "PASS"]
    if bad:
        lines += ["", "## PASS 가 아닌 run", "", "| run | status | max|Δ| | 상세 |", "|---|---|---|---|"]
        for r in bad:
            detail = r.get("note") or r.get("error") or r.get("load_error") or ""
            if r.get("diffs") and r["status"] == "MISMATCH":
                detail = "; ".join(f"{k}: {v['expected']:.2f}→{v['evaluated']:.2f}" for k, v in r["diffs"].items() if isinstance(v, dict))
            md = "" if r.get("max_abs_diff") is None else f"{r['max_abs_diff']:.3f}"
            lines.append(f"| {r['run']} | {r['status']} | {md} | {detail[:160]} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=str(ROOT / "docs" / "results_master.csv"))
    ap.add_argument("--out", required=True, help="결과 디렉터리 (results-<shard>.jsonl, results.csv, summary.md)")
    ap.add_argument("--sandbox", default=None, help="기본 <out>/sandbox — 프로세스 CWD 가 여기로 바뀐다")
    ap.add_argument("--group", action="append")
    ap.add_argument("--dataset", choices=["icbhi", "sprsound"])
    ap.add_argument("--run", action="append")
    ap.add_argument("--shard", default=None, help="i/n — 행을 i::n 으로 나눠 맡는다 (GPU 병렬용)")
    ap.add_argument("--smoke", action="store_true", help="그룹·출처별 1개씩만")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.05, help="PASS 허용 |Δ| (% 단위 지표)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-temp", type=int, default=74,
                    help="run 사이 온도 게이트(°C): 이 값 이상이면 5°C 아래로 식을 때까지 대기. 0 이면 끔")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(out_dir)
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    for noisy in ("lightning", "lightning.pytorch", "lightning.fabric", "rsc_toolkit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # run 당 수십 줄의 INFO 를 막는다
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _install_readonly_chain_guard()
    _sb = pathlib.Path(args.sandbox).resolve() if args.sandbox else out_dir / "sandbox"
    if (ROOT / "outputs") in _sb.parents:
        raise SystemExit(f"샌드박스 {_sb} 가 outputs/ 안이다 — 콜백이 쓰는 best.json 이 results_master 에 섞인다. --sandbox 를 밖으로 두라.")
    sandbox = pathlib.Path(args.sandbox).resolve() if args.sandbox else out_dir / "sandbox"
    make_sandbox(sandbox)
    os.chdir(sandbox)  # outputs/ 로 향하는 어떤 상대 경로 쓰기도 여기로 떨어진다

    args.master_rows = list(csv.DictReader(open(args.master)))
    rows = select_rows(args)
    tag = (args.shard or "0/1").replace("/", "of")
    out_path = out_dir / f"results-{tag}.jsonl"
    log.info("%d runs → %s (sandbox=%s)", len(rows), out_path, sandbox)
    with open(out_path, "a") as fh:
        for i, row in enumerate(rows, 1):
            waited = wait_for_cool_gpu(args.max_temp, args.max_temp - 5)
            if waited:
                log.info("GPU 온도 게이트: %.0fs 대기 후 재개", waited)
            rec = run_one(row, args, sandbox)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            md = rec.get("max_abs_diff")
            log.info("[%d/%d] %-42s %-9s |Δ|=%s  %ss  %s", i, len(rows), rec["run"], rec["status"],
                     "-" if md is None else f"{md:.3f}", rec.get("seconds"), (rec.get("note") or rec.get("error") or "")[:120])
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402  (모듈 최상단 import 는 --summarize 를 느리게 한다)

    raise SystemExit(main())
