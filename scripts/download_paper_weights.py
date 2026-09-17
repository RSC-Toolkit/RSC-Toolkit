# -*- coding: utf-8 -*-
"""
공개된 논문 weight 내려받기 — weights_manifest.csv 의 url 로 묶음(best.pth·best.json·.hydra/·sha256.txt)을 받고
sha256.txt 의 모든 항목과 대장의 best_pth_sha256 을 대조한다. url 은 Google Drive **폴더 링크**(…/folders/<id>)로,
run 폴더 자체이거나 여러 run 폴더를 담은 **상위 폴더**여도 된다(상위 폴더면 <run>/ 아래 파일만 골라 받는다).

  python scripts/download_paper_weights.py --list
  python scripts/download_paper_weights.py --run icbhi-cnn6-ce-s1 --out paper_weights/

url 이 아직 "[Google Drive]" 자리표시자면 안내만 출력하고 종료 코드 2 를 돌려준다(업로드 전). 파일 하나 링크로는 묶음이 안 된다.
받은 뒤 재평가:
  rsc-toolkit-eval paper_weights/icbhi-cnn6-ce-s1 --project-root <clone> --override data.params.data_dir=<dataset dir>
종료 코드: 0 전부 OK · 1 오류(대장에 없음·해시 불일치·다운로드 실패) · 2 링크 미공개 (여러 run 이면 가장 큰 값).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights_manifest.csv"
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


def _folder_id(url: str) -> str:
    tail = url.split("/folders/", 1)[1]
    return tail.split("?", 1)[0].split("/", 1)[0].strip()


def download_run(url: str, run: str, dst: pathlib.Path) -> str | None:
    """Drive 폴더 링크에서 run 묶음을 dst/ 에 받는다. 문제가 있으면 사유를 돌려준다.

    폴더 목록은 gdown 의 skip_download 로 먼저 읽는다 — url 이 run 폴더면 전부, 상위 폴더면 `<run>/` 아래 파일만 받는다
    (gdown --folder 는 하위 폴더 하나만 고를 수 없어 파일 단위로 받는다).
    """
    import os

    import gdown

    listing_base = dst.parent.resolve()
    try:
        files = gdown.download_folder(id=_folder_id(url), output=str(listing_base) + os.sep, skip_download=True, quiet=True)
    except Exception as exc:                        # 권한(링크 공유 아님)·네트워크
        return f"폴더 목록을 읽지 못했다: {type(exc).__name__}: {str(exc)[:200]}"
    if not files:
        return "폴더가 비어 있다 (아직 업로드 전이거나 링크가 다른 폴더다)"
    entries = []
    for f in files:
        try:
            rel = pathlib.Path(f.path).resolve().relative_to(listing_base).parts
        except ValueError:
            rel = pathlib.Path(f.path).parts
        entries.append((f.id, tuple(rel)))
    # 어느 깊이에 있든 `<run>/` 디렉터리 아래 파일을 고른다 (상위 폴더 안에 또 폴더가 있어도 된다)
    wanted = [(fid, parts[parts.index(run) + 1:]) for fid, parts in entries if run in parts[:-1]]
    if not wanted:
        # url 자체가 run 폴더: best.pth 가 하나뿐이면 그 디렉터리를 묶음 루트로 본다
        roots = {parts[:-1] for _, parts in entries if parts[-1] == "best.pth"}
        if len(roots) != 1:
            seen = sorted({"/".join(parts) for _, parts in entries})[:8]
            return f"폴더에 {run}/ 가 없고 best.pth 도 하나가 아니다 (있는 것: {seen})"
        prefix = roots.pop()
        wanted = [(fid, parts[len(prefix):]) for fid, parts in entries if parts[:len(prefix)] == prefix]
    dst.mkdir(parents=True, exist_ok=True)
    for fid, parts in wanted:
        if not parts:
            continue
        local = dst.joinpath(*parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        got = gdown.download(id=fid, output=str(local), quiet=True)
        if not got:
            return f"{'/'.join(parts)} 다운로드 실패"
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
            print(f"{run}: 아직 공개 링크가 없다(대장 url={r['url']!r}). 링크가 생기면 weights_manifest.csv 의 url 을 채우고 다시 실행하라. "
                  f"묶음이 손에 있으면 {dst}/ 에 직접 두어도 된다.")
            rc = max(rc, 2); continue
        if "/folders/" not in r["url"]:
            print(f"{run}: url 은 Google Drive 폴더 링크(…/folders/<id>)여야 한다: {r['url']}"); rc = max(rc, 1); continue
        try:
            import gdown  # noqa: F401
        except ImportError:
            raise SystemExit("gdown 이 필요하다: pip install gdown (또는 pip install -e '.[notebook]')")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"{run}: {r['url']} → {dst}")
        problem = download_run(r["url"], run, dst)
        if problem:
            print(f"{run}: 다운로드 실패 — {problem}"); rc = max(rc, 1); continue
        problem = verify_bundle(dst, r["best_pth_sha256"])
        if problem:
            print(f"{run}: 검증 실패 — {problem}"); rc = max(rc, 1)
        else:
            print(f"{run}: OK → {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
