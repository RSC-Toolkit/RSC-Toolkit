# -*- coding: utf-8 -*-
"""
공개된 논문 weight 내려받기 — docs/paper/weights_manifest.csv 의 url 로 run 묶음(best.pth·best.json·.hydra/)을 받고 sha256 을 대조한다.

  python scripts/download_paper_weights.py --list
  python scripts/download_paper_weights.py --run icbhi-cnn6-ce-s1 --out paper_weights/

url 이 아직 "[Google Drive]" 자리표시자면 안내만 출력하고 종료 코드 2 를 돌려준다(업로드 전). Google Drive 폴더/파일 링크는 gdown 으로 받는다.
받은 뒤 재평가: rsc-toolkit-eval paper_weights/icbhi-cnn6-ce-s1 --project-root <clone> --override data.params.data_dir=<dataset dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "paper" / "weights_manifest.csv"
URL_PLACEHOLDER = "[Google Drive]"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--out", default="paper_weights")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    rows = {r["run"]: r for r in csv.DictReader(open(MANIFEST))}
    if args.list or not args.run:
        for r in rows.values():
            print(f"{r['run']:28s} {r['dataset']:8s} Score {float(r['icbhi_score']):6.2f}  {int(r['best_pth_bytes'])/1e6:6.1f} MB  {r['url']}")
        return 0
    rc = 0
    for run in args.run:
        r = rows.get(run)
        if r is None:
            print(f"{run}: 대장에 없다"); rc = 1; continue
        dst = pathlib.Path(args.out) / run
        if (dst / "best.pth").exists() and sha256(dst / "best.pth") == r["best_pth_sha256"]:
            print(f"{run}: 이미 있음, sha256 일치 → {dst}"); continue
        if r["url"] == URL_PLACEHOLDER or not r["url"]:
            print(f"{run}: 아직 공개 링크가 없다(대장 url={r['url']!r}). 링크가 생기면 docs/paper/weights_manifest.csv 의 url 을 채우고 다시 실행하라.")
            rc = 2; continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            import gdown  # noqa: F401
        except ImportError:
            raise SystemExit("pip install gdown 이 필요하다")
        cmd = [sys.executable, "-m", "gdown", "--folder", r["url"], "-O", str(dst)] if "/folders/" in r["url"] else \
              [sys.executable, "-m", "gdown", r["url"], "-O", str(dst) + "/"]
        print("$", " ".join(cmd)); subprocess.run(cmd, check=True)
        got = sha256(dst / "best.pth") if (dst / "best.pth").exists() else None
        if got != r["best_pth_sha256"]:
            print(f"{run}: sha256 불일치 또는 best.pth 없음 (got {got})"); rc = 1
        else:
            print(f"{run}: OK → {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
