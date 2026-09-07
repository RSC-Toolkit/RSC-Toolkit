# -*- coding: utf-8 -*-
"""
논문 OOD 수치(outputs/<group>/<run>/ood.json) 재검증 — QLung·BTS-CARD.

ood.json 은 같은 best.pth 를 SPRSound OOD 프로토콜로 평가한 산출물이라(scripts/reproduction/eval_qlung_ood.py,
eval_btscard_checkpoint.py), 그 스크립트를 **출력만 샌드박스로 돌려** 다시 실행하고 기존 ood.json 과
(S_p, S_e, icbhi_score) 를 경로별로 대조한다. outputs/ 는 읽기만 한다.

  .venv/bin/python scripts/verify_paper_ood.py --out DIR [--group qlung|btscard] [--run NAME ...] [--dry-run] [--tol 0.05]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
METRIC_KEYS = ("S_p", "S_e", "icbhi_score")


def wait_for_cool_gpu(max_temp: int, resume_temp: int) -> None:
    """run 사이 온도 게이트 — verify_paper_weights.py 와 같은 규칙 (서버 규칙: 80°C 초과 금지)."""
    import os
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if max_temp <= 0 or not idx.isdigit():
        return
    gate_open = True
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            temp = int(out.splitlines()[0])
        except Exception:
            return
        if (gate_open and temp < max_temp) or (not gate_open and temp <= resume_temp):
            return
        gate_open = False
        time.sleep(15)


def command_for(row: dict, run_dir: pathlib.Path, old: dict, out_json: pathlib.Path, device: str) -> list[str]:
    ckpt = str(run_dir / "best.pth")
    if row["paper_group"] == "qlung":
        scopes = ",".join(k for k in ("inter", "all") if k in old)
        return [PY, "scripts/reproduction/eval_qlung_ood.py", "--ckpt", ckpt, "--out", str(out_json),
                "--scopes", scopes or "inter,all", "--device", device]
    if row["paper_group"] == "btscard":
        srcs = [k[len("sprsound_"):] for k in old if k.startswith("sprsound_")]
        cmd = [PY, "scripts/reproduction/eval_btscard_checkpoint.py", "--ckpt", ckpt, "--ckpt_format",
               old.get("ckpt_format", "toolkit"), "--sprsound", ",".join(srcs) or "author", "--out", str(out_json),
               "--device", device]
        if old.get("amp"):
            cmd.append("--amp")
        if "icbhi" not in old:
            cmd.append("--no_icbhi")
        return cmd
    raise ValueError(f"OOD 프로토콜을 모르는 그룹: {row['paper_group']}")


def walk_metrics(obj, path=()):
    """ood.json 안에서 (S_p, S_e, icbhi_score) 를 가진 dict 를 전부 찾아 경로와 함께 낸다."""
    if isinstance(obj, dict):
        if all(k in obj for k in METRIC_KEYS):
            yield path, {k: float(obj[k]) for k in METRIC_KEYS}
        for k, v in obj.items():
            if k in ("ckpt_info", "icbhi_best", "reference_table2"):  # 학습 best/논문값 — 재평가 대상이 아니다
                continue
            yield from walk_metrics(v, path + (k,))


def get_path(obj, path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=str(ROOT / "docs" / "results_master.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", choices=["qlung", "btscard"])
    ap.add_argument("--run", action="append")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-temp", type=int, default=74, help="run 사이 GPU 온도 게이트(°C), 0 이면 끔")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(args.master)) if r["has_ood"] == "1"]
    if args.group:
        rows = [r for r in rows if r["paper_group"] == args.group]
    if args.run:
        rows = [r for r in rows if r["run_name"] in set(args.run)]

    results = []
    for i, row in enumerate(rows, 1):
        run_dir = ROOT / row["run_dir"]
        old = json.loads((run_dir / "ood.json").read_text())
        out_json = out_dir / "ood" / row["run_name"] / "ood.json"
        cmd = command_for(row, run_dir, old, out_json, args.device)
        rec = {"group": row["paper_group"], "run": row["run_name"], "cmd": " ".join(cmd)}
        if args.dry_run:
            print(f"[{i}/{len(rows)}] {row['run_name']}: {' '.join(cmd)}")
            continue
        wait_for_cool_gpu(args.max_temp, args.max_temp - 5)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rec["seconds"] = round(time.time() - t0, 1)
        if proc.returncode != 0 or not out_json.exists():
            rec["status"] = "ERROR"
            rec["error"] = (proc.stderr or proc.stdout)[-1500:]
        else:
            new = json.loads(out_json.read_text())
            diffs, worst, missing = {}, 0.0, []
            for path, exp in walk_metrics(old):
                got = get_path(new, path)
                if not isinstance(got, dict) or not all(k in got for k in METRIC_KEYS):
                    missing.append("/".join(path))
                    continue
                d = {k: abs(exp[k] - float(got[k])) for k in METRIC_KEYS}
                diffs["/".join(path)] = {"expected": exp, "evaluated": {k: float(got[k]) for k in METRIC_KEYS}, "max_diff": max(d.values())}
                worst = max(worst, max(d.values()))
            rec.update({"status": "PASS" if diffs and worst <= args.tol and not missing else ("MISMATCH" if diffs else "ERROR"),
                        "max_abs_diff": worst, "n_compared": len(diffs), "missing_paths": missing, "diffs": diffs})
        results.append(rec)
        print(f"[{i}/{len(rows)}] {row['run_name']:36s} {rec['status']:8s} |Δ|={rec.get('max_abs_diff', float('nan')):.3f} "
              f"n={rec.get('n_compared', 0)} {rec['seconds']}s {rec.get('error', '')[-200:]}", flush=True)
        with open(out_dir / "ood_results.jsonl", "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if not args.dry_run:
        counts = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print("OOD 검증 요약:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
