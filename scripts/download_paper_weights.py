# -*- coding: utf-8 -*-
"""
공개된 논문 weight 내려받기 — docs/paper/weights_manifest.csv 의 url(**run 폴더 링크**)로 묶음(best.pth·best.json·.hydra/·sha256.txt)을 받고
sha256.txt 의 모든 항목과 대장의 best_pth_sha256 을 대조한다.

  python scripts/download_paper_weights.py --list
  python scripts/download_paper_weights.py --run icbhi-cnn6-ce-s1 --out paper_weights/

url 이 아직 "[Google Drive]" 자리표시자면 안내만 출력하고 종료 코드 2 를 돌려준다(업로드 전). Google Drive 는 **폴더 링크**(…/folders/<id>)만
받는다(gdown --folder; 파일 하나 링크로는 묶음이 안 된다). 받은 뒤 재평가:
  rsc-toolkit-eval paper_weights/icbhi-cnn6-ce-s1 --project-root <clone> --override data.params.data_dir=<dataset dir>
종료 코드: 0 전부 OK · 1 오류(대장에 없음·해시 불일치·다운로드 실패) · 2 링크 미공개 (여러 run 이면 가장 큰 값).
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


def verify_bundle(dst: pathlib.Path, expected_best_sha: str) -> str | None:
    """sha256.txt 의 전 항목 + 대장의 best.pth 해시를 대조. 문제가 있으면 사유를 돌려준다."""
    if not (dst / "best.pth").exists():
        return "best.pth 없음"
    if sha256(dst / "best.pth") != expected_best_sha:
        return "best.pth sha256 이 대장과 다르다"
    listing = dst / "sha256.txt"
    if not listing.exists():
        return "sha256.txt 없음 (묶음이 불완전하다)"
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        f = dst / rel
        if not f.exists():
            return f"{rel} 없음"
        if sha256(f) != digest:
            return f"{rel} sha256 불일치"
    return None


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
            print(f"{run}: 대장에 없다"); rc = max(rc, 1); continue
        dst = pathlib.Path(args.out) / run
        if dst.exists():
            problem = verify_bundle(dst, r["best_pth_sha256"])
            if problem is None:
                print(f"{run}: 이미 있음, sha256 전부 일치 → {dst}"); continue
            print(f"{run}: 기존 폴더 {dst} 가 대장과 다르다 ({problem}) — 다시 받는다")
        if r["url"] == URL_PLACEHOLDER or not r["url"]:
            print(f"{run}: 아직 공개 링크가 없다(대장 url={r['url']!r}). 링크가 생기면 docs/paper/weights_manifest.csv 의 url 을 채우고 다시 실행하라. "
                  f"묶음이 손에 있으면 {dst}/ 에 직접 두어도 된다.")
            rc = max(rc, 2); continue
        if "/folders/" not in r["url"]:
            print(f"{run}: url 은 Google Drive 폴더 링크(…/folders/<id>)여야 한다: {r['url']}"); rc = max(rc, 1); continue
        try:
            import gdown  # noqa: F401
        except ImportError:
            raise SystemExit("gdown 이 필요하다: pip install gdown (또는 pip install -e '.[notebook]')")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "gdown", "--folder", r["url"], "-O", str(dst)]
        print("$", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"{run}: 다운로드 실패 (exit {exc.returncode})"); rc = max(rc, 1); continue
        problem = verify_bundle(dst, r["best_pth_sha256"])
        if problem:
            print(f"{run}: 검증 실패 — {problem}"); rc = max(rc, 1)
        else:
            print(f"{run}: OK → {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
