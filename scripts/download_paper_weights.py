# -*- coding: utf-8 -*-
"""
Download published paper weights — fetches the bundle (best.pth, best.json, .hydra/, sha256.txt) from the url in weights_manifest.csv
and checks every entry of sha256.txt plus best_pth_sha256 from the manifest. The url is a Google Drive **folder link** (…/folders/<id>);
it may be the run folder itself or a **parent folder** holding several run folders (then only files under <run>/ are downloaded).

  python scripts/download_paper_weights.py --list
  python scripts/download_paper_weights.py --run icbhi-cnn6-ce-s1 --out paper_weights/

If the url is still the "[Google Drive]" placeholder, a notice is printed and exit code 2 is returned (not uploaded yet). A single-file link cannot hold a bundle.
Re-evaluation after download:
  rsc-toolkit-eval paper_weights/icbhi-cnn6-ce-s1 --project-root <clone> --override data.params.data_dir=<dataset dir>
Exit codes: 0 all OK · 1 error (not in manifest, hash mismatch, download failure) · 2 link not published (the largest value over several runs).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights_manifest.csv"
ATTEMPTS = 3          # Google Drive requests are retried this many times
RETRY_WAIT_S = 20
URL_PLACEHOLDER = "[Google Drive]"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(dst: pathlib.Path, expected_best_sha: str) -> str | None:
    """Check every entry of sha256.txt plus the best.pth hash from the manifest. Returns the reason if there is a problem."""
    if not (dst / "best.pth").exists():
        return "best.pth missing"
    if sha256(dst / "best.pth") != expected_best_sha:
        return "best.pth sha256 differs from the manifest"
    listing = dst / "sha256.txt"
    if not listing.exists():
        return "sha256.txt missing (incomplete bundle)"
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        f = dst / rel
        if not f.exists():
            return f"{rel} missing"
        if sha256(f) != digest:
            return f"{rel} sha256 mismatch"
    return None


def _folder_id(url: str) -> str:
    tail = url.split("/folders/", 1)[1]
    return tail.split("?", 1)[0].split("/", 1)[0].strip()


def select_bundle_files(entries: list[tuple[str, tuple[str, ...]]], run: str) -> tuple[list, str | None]:
    """Pick the files of one run bundle from a Drive folder listing.

    entries: (file id, path parts relative to the listing root). A bundle root is a directory that directly contains best.pth.
    The run's bundle is the root whose last component is `run`; files of deeper bundles inside it are excluded (the family folder
    of a seed-1 run can have the same name as the run, e.g. icbhi-patchmix-ce/icbhi-patchmix-ce{,-s2,...}/). If the listing holds
    exactly one bundle, that bundle is used (the url is the run folder itself). Returns ([(file id, parts inside the bundle)], problem).
    """
    roots = {parts[:-1] for _, parts in entries if parts and parts[-1] == "best.pth"}
    named = [r for r in roots if r and r[-1] == run]
    if len(named) == 1:
        prefix = named[0]
    elif not named and len(roots) == 1:
        prefix = next(iter(roots))
    else:
        seen = sorted({"/".join(parts) for _, parts in entries})[:8]
        return [], f"cannot find a single bundle for {run} in the folder (bundles named {run}: {len(named)}; found: {seen})"
    deeper = [r for r in roots if len(r) > len(prefix) and r[:len(prefix)] == prefix]
    wanted = [(fid, parts[len(prefix):]) for fid, parts in entries
              if parts[:len(prefix)] == prefix and not any(parts[:len(r)] == r for r in deeper)]
    return wanted, None


def download_run(url: str, run: str, dst: pathlib.Path) -> str | None:
    """Download a run bundle from a Drive folder link into dst/. Returns the reason if there is a problem.

    The folder listing is read first with gdown's skip_download — if the url is the run folder, everything is downloaded; if it is a
    parent folder, only files under `<run>/` (gdown --folder cannot select a single subfolder, so files are downloaded one by one).
    """
    import os

    import gdown

    listing_base = dst.parent.resolve()
    files, err = None, None
    for attempt in range(1, ATTEMPTS + 1):          # Google Drive sometimes answers a large folder listing with HTTP 500
        try:
            files = gdown.download_folder(id=_folder_id(url), output=str(listing_base) + os.sep, skip_download=True, quiet=True)
            break
        except Exception as exc:                    # permissions (link not shared) or network
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt < ATTEMPTS:
                print(f"{run}: reading the folder listing failed (attempt {attempt}/{ATTEMPTS}); retrying in {RETRY_WAIT_S} s")
                time.sleep(RETRY_WAIT_S)
    if files is None:
        return f"failed to read the folder listing after {ATTEMPTS} attempts: {err}"
    if not files:
        return "folder is empty (not uploaded yet, or the link points to a different folder)"
    entries = []
    for f in files:
        try:
            rel = pathlib.Path(f.path).resolve().relative_to(listing_base).parts
        except ValueError:
            rel = pathlib.Path(f.path).parts
        entries.append((f.id, tuple(rel)))
    wanted, problem = select_bundle_files(entries, run)
    if problem:
        return problem
    dst.mkdir(parents=True, exist_ok=True)
    for fid, parts in wanted:
        if not parts:
            continue
        local = dst.joinpath(*parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        got = None
        for attempt in range(1, ATTEMPTS + 1):
            got = gdown.download(id=fid, output=str(local), quiet=True)
            if got:
                break
            if attempt < ATTEMPTS:
                time.sleep(RETRY_WAIT_S)
        if not got:
            return f"{'/'.join(parts)} download failed"
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
            print(f"{run}: not in the manifest"); rc = max(rc, 1); continue
        dst = pathlib.Path(args.out) / run
        if dst.exists():
            problem = verify_bundle(dst, r["best_pth_sha256"])
            if problem is None:
                print(f"{run}: already present, all sha256 match → {dst}"); continue
            print(f"{run}: existing folder {dst} differs from the manifest ({problem}) — downloading again")
        if r["url"] == URL_PLACEHOLDER or not r["url"]:
            print(f"{run}: no public link yet (manifest url={r['url']!r}). Once a link exists, fill in the url in weights_manifest.csv and run again. "
                  f"If you already have the bundle, you can place it in {dst}/ directly.")
            rc = max(rc, 2); continue
        if "/folders/" not in r["url"]:
            print(f"{run}: url must be a Google Drive folder link (…/folders/<id>): {r['url']}"); rc = max(rc, 1); continue
        try:
            import gdown  # noqa: F401
        except ImportError:
            raise SystemExit("gdown is required: pip install gdown (or pip install -e '.[notebook]')")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"{run}: {r['url']} → {dst}")
        problem = download_run(r["url"], run, dst)
        if problem:
            print(f"{run}: download failed — {problem}"); rc = max(rc, 1); continue
        problem = verify_bundle(dst, r["best_pth_sha256"])
        if problem:
            print(f"{run}: verification failed — {problem}"); rc = max(rc, 1)
        else:
            print(f"{run}: OK → {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
