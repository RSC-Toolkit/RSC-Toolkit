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

# 설정 복구·정제·로드·평가·대조의 본체는 rsc_toolkit.evaluate 로 승격했다 (리팩토링 P2-3).
# 이 스크립트는 논문 weight **전수** 검증의 드라이버(행 선택·샤딩·온도 게이트·MVST 융합·요약)만 남긴다.
# 아래 이름들은 scripts/eval_ood_sprsound.py 가 `vw.<이름>` 으로 쓰므로 유지한다 (ROOT 를 묶은 얇은 창구).
from rsc_toolkit import evaluate as ev  # noqa: E402
from rsc_toolkit.evaluate import (  # noqa: E402,F401
    ABS_PREFIXES, KEEP_CALLBACKS, READ_ONLY_ROOTS, STD_KEYS, VOTE_MAP,
    adapt_buffer_shapes as _adapt_buffer_shapes, build_and_validate, compare, load_log_config,
)

PAPER_KEYS = ("paper_S_p", "paper_S_e", "paper_score", "paper_acc")

log = logging.getLogger("verify")


def load_snapshot_config(hydra_dir: pathlib.Path) -> dict:
    return ev.load_snapshot_config(hydra_dir, ROOT)


def find_run_log(run_name: str) -> pathlib.Path | None:
    return ev.find_run_log(run_name, ROOT)


def _absolutize(obj, skip_path=()):
    return ev.absolutize(obj, ROOT, skip_path)


def sanitize_config(cfg: dict, sandbox: pathlib.Path, run_name: str) -> tuple[dict, list[str]]:
    return ev.sanitize_config(cfg, sandbox, run_name, ROOT)


CHAIN_NOTES: list = []


def _install_readonly_chain_guard() -> None:
    """캐시 체인 대장을 읽기 전용으로 (프로세스 전체). 어긋남은 CHAIN_NOTES 에 쌓인다."""
    ev.install_readonly_chain_guard(CHAIN_NOTES)



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
    # 스냅샷·로그가 모두 없는 정규 seed run(예: icbhi-m2d-base-s1, sprs-operact-ft-s1 — 초기 배치의 hydra 디렉터리 유실):
    # seed·run.name 외의 설정은 시리즈 안에서 같으므로 다른 seed 의 스냅샷으로 재구성한다. 출처(sibling:…)를 남긴다.
    m = re.match(r"^(.*)-s(\d+)$", row["run_name"])
    if m and master_rows is not None:
        for k in range(1, 6):
            cand = f"{m.group(1)}-s{k}"
            if cand == row["run_name"]:
                continue
            for r in master_rows:
                if r["run_name"] == cand and r["provenance"] == "hydra" and r.get("hydra_dir"):
                    return load_snapshot_config(ROOT / r["hydra_dir"]), f"sibling:{cand}"
    return None, "none"


def make_sandbox(path: pathlib.Path) -> None:
    ev.make_sandbox(path, ROOT)



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
