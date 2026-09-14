# -*- coding: utf-8 -*-
"""
논문 weight 공개 묶음 만들기 + 공개 대장(docs/paper/weights_manifest.csv) 갱신.

run 하나당 폴더 하나: best.pth · best.json · .hydra/config.yaml(+overrides.yaml) · README.txt · sha256.txt
→ 이 폴더를 그대로 `rsc-toolkit-eval <폴더> --project-root <clone>` 로 재평가할 수 있다(설정 스냅샷이 동봉되므로 outputs/ 가 없어도 된다).

  .venv/bin/python scripts/paper/bundle_weights.py --run icbhi-cnn6-ce-s1 [--run …] --out outputs/weights_release
  .venv/bin/python scripts/paper/bundle_weights.py --family icbhi-cnn6-ce --out outputs/weights_release     # seed 1–5

대장 열: run, family, dataset, model, config_name, S_p, S_e, icbhi_score, epoch, best_pth_sha256, best_pth_bytes, bundle_sha256, url
url 은 업로드 전에는 "[Google Drive]" 자리표시자이고, 업로드 뒤 사용자가 링크로 바꾼다(scripts/download_paper_weights.py 가 읽는다).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs" / "results_master.csv"
MANIFEST = ROOT / "docs" / "paper" / "weights_manifest.csv"
URL_PLACEHOLDER = "[Google Drive]"
FIELDS = ["run", "family", "dataset", "model", "config_name", "S_p", "S_e", "icbhi_score", "epoch",
          "best_pth_sha256", "best_pth_bytes", "bundle_sha256", "url"]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_of(run: str) -> str:
    return re.sub(r"-s\d+$", "", run)


def bundle(row: dict, out: pathlib.Path) -> dict:
    run = row["run_name"]; run_dir = ROOT / row["run_dir"]; hydra_dir = ROOT / row["hydra_dir"] if row.get("hydra_dir") else None
    dst = out / run
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("best.pth", "best.json"):
        shutil.copy2(run_dir / name, dst / name)
    if hydra_dir is not None and (hydra_dir / ".hydra" / "config.yaml").exists():
        (dst / ".hydra").mkdir()
        for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
            if (hydra_dir / ".hydra" / name).exists():
                shutil.copy2(hydra_dir / ".hydra" / name, dst / ".hydra" / name)
    else:
        raise SystemExit(f"{run}: hydra 스냅샷이 없어 묶음을 만들 수 없다 (results_master hydra_dir={row.get('hydra_dir')!r})")
    best = json.loads((dst / "best.json").read_text())
    (dst / "README.txt").write_text(
        f"RSC-Toolkit paper weight: {run}\n"
        f"family={family_of(run)} dataset={row['dataset']} model={row['model_name']} config={row['config_name']}\n"
        f"best.json: S_p={best.get('S_p'):.2f} S_e={best.get('S_e'):.2f} Score={best.get('icbhi_score'):.2f} (epoch {best.get('epoch')})\n\n"
        "Files: best.pth (model state_dict under key 'model'), best.json (metrics at the saved epoch), .hydra/config.yaml (training config snapshot).\n"
        "Re-evaluate (from a git clone of RSC-Toolkit, dataset downloaded and indexed):\n"
        f"  rsc-toolkit-eval <this folder> --project-root <clone> --override data.params.data_dir=<dataset dir>\n"
        "See docs/TUTORIAL.md.\n")
    files = sorted(p for p in dst.rglob("*") if p.is_file() and p.name != "sha256.txt")
    lines = [f"{sha256(p)}  {p.relative_to(dst).as_posix()}" for p in files]
    (dst / "sha256.txt").write_text("\n".join(lines) + "\n")
    return {"run": run, "family": family_of(run), "dataset": row["dataset"], "model": row["model_name"], "config_name": row["config_name"],
            "S_p": f"{best.get('S_p'):.4f}", "S_e": f"{best.get('S_e'):.4f}", "icbhi_score": f"{best.get('icbhi_score'):.4f}", "epoch": best.get("epoch"),
            "best_pth_sha256": sha256(dst / "best.pth"), "best_pth_bytes": (dst / "best.pth").stat().st_size,
            "bundle_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(), "url": URL_PLACEHOLDER}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--family", action="append", default=[], help="family 이름 → seed 1–5 run 전부")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "weights_release"))
    args = ap.parse_args()
    master = {r["run_name"]: r for r in csv.DictReader(open(MASTER))}
    runs = list(args.run) + [r for f in args.family for r in master if family_of(r) == f and re.search(r"-s[1-5]$", r)]
    if not runs:
        ap.error("--run 또는 --family 를 주어라")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = {r["run"]: r for r in csv.DictReader(open(MANIFEST))} if MANIFEST.exists() else {}
    for run in runs:
        if run not in master:
            raise SystemExit(f"{run} 이 results_master 에 없다")
        rec = bundle(master[run], out)
        if run in rows and rows[run].get("url") not in ("", URL_PLACEHOLDER):
            rec["url"] = rows[run]["url"]          # 이미 올린 링크는 보존
        rows[run] = rec
        print(f"{run:28s} → {out / run}  best.pth {rec['best_pth_bytes']/1e6:.1f} MB  sha256 {rec['best_pth_sha256'][:12]}…")
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for run in sorted(rows):
            w.writerow({k: rows[run].get(k, "") for k in FIELDS})
    print(f"manifest: {MANIFEST} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
