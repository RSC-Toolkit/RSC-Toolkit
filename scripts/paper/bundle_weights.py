# -*- coding: utf-8 -*-
"""
논문 weight 공개 묶음 만들기 + 공개 대장(weights_manifest.csv) 갱신.

run 하나당 폴더 하나: best.pth · best.json · .hydra/config.yaml(+overrides.yaml) · README.txt · sha256.txt
→ 이 폴더를 그대로 `rsc-toolkit-eval <폴더> --project-root <clone>` 로 재평가할 수 있다(설정 스냅샷이 동봉되므로 outputs/ 가 없어도 된다).

  .venv/bin/python scripts/paper/bundle_weights.py --run icbhi-cnn6-ce-s1 [--run …] --out outputs/weights_release
  .venv/bin/python scripts/paper/bundle_weights.py --family icbhi-cnn6-ce --out outputs/weights_release     # seed 1–5
  .venv/bin/python scripts/paper/bundle_weights.py --all-paper --link --layout nested --out outputs/weights_release
      # 표 1·2 의 family 전부(anchors.csv; M2D 계열은 --include-m2d 없으면 제외) → <out>/<dataset>/<family>/<run>/ ;
      # --link 는 best.pth 를 복사 대신 하드링크(같은 파일시스템, 추가 용량 0). INDEX.tsv·README.md 도 만든다.

hydra 스냅샷이 없는 run(초기 배치·로그 복구·MVST 융합)은 scripts/verify_paper_weights.resolve_config 와 같은 규칙으로
설정을 재구성해 .hydra/config.yaml 로 넣고 출처를 .hydra/SNAPSHOT_SOURCE.txt 에 남긴다(재구성 불가 = MVST 융합 MLP 는 .hydra 없이).

대장 열: run, family, dataset, model, config_name, S_p, S_e, icbhi_score, epoch, best_pth_sha256, best_pth_bytes, bundle_sha256, url
url 은 업로드 전에는 "[Google Drive]" 자리표시자이고, 업로드 뒤 사용자가 **run 폴더 링크**로 바꾼다(scripts/download_paper_weights.py 가 읽는다).
안전장치: 대상 run 전부를 먼저 검사(best.pth·best.json·hydra 스냅샷)하고, 묶음은 임시 폴더에 만든 뒤 교체하며, 대장은 run 마다 갱신한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs" / "results_master.csv"
MANIFEST = ROOT / "weights_manifest.csv"
URL_PLACEHOLDER = "[Google Drive]"
FIELDS = ["run", "family", "dataset", "model", "config_name", "S_p", "S_e", "icbhi_score", "epoch",
          "best_pth_sha256", "best_pth_bytes", "bundle_sha256", "url"]
SNAPSHOT_FILES = ("config.yaml", "overrides.yaml")      # hydra.yaml 은 개발 머신 절대 경로가 들어 있고 평가에 안 쓰여 동봉하지 않는다


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_of(run: str) -> str:
    return re.sub(r"-s\d+$", "", run)


def dataset_of(row: dict) -> str:
    if row.get("dataset"):
        return row["dataset"]
    head = row["run_name"].split("-", 1)[0]                     # icbhi-… / sprs-…
    return {"icbhi": "icbhi", "sprs": "sprsound"}.get(head, "")


def check_row(row: dict) -> tuple[pathlib.Path, pathlib.Path]:
    """묶음에 필요한 입력이 전부 있는지 먼저 본다. (run_dir, hydra_dir) 를 돌려준다."""
    run = row["run_name"]; run_dir = ROOT / row["run_dir"]
    for name in ("best.pth", "best.json"):
        if not (run_dir / name).exists():
            raise SystemExit(f"{run}: {run_dir / name} 이 없다")
    hydra_dir = ROOT / row["hydra_dir"] if row.get("hydra_dir") else None
    if hydra_dir is not None and not (hydra_dir / ".hydra" / "config.yaml").exists():
        hydra_dir = None                                     # 스냅샷이 없으면 bundle() 이 verify_paper_weights.resolve_config 로 재구성한다
    if not dataset_of(row):
        raise SystemExit(f"{run}: dataset 을 알 수 없다 (results_master dataset 열 비어 있음, run 이름 접두사도 아님)")
    return run_dir, hydra_dir


_VW = None


def _vw():
    """scripts/verify_paper_weights.py 를 모듈로 (스냅샷 재구성 규칙을 한 곳에 둔다)."""
    global _VW
    if _VW is None:
        spec = importlib.util.spec_from_file_location("verify_paper_weights", ROOT / "scripts" / "verify_paper_weights.py")
        _VW = importlib.util.module_from_spec(spec); spec.loader.exec_module(_VW)
    return _VW


def _place(src: pathlib.Path, dst: pathlib.Path, link: bool) -> None:
    if link:
        try:
            os.link(src, dst); return
        except OSError:                                      # 다른 파일시스템 등 — 복사로
            pass
    shutil.copy2(src, dst)


def bundle(row: dict, out: pathlib.Path, master_rows: list[dict] | None = None, link: bool = False) -> dict:
    run = row["run_name"]; run_dir, hydra_dir = check_row(row)
    dst = out / run; tmp = out / f".tmp-{run}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _place(run_dir / "best.pth", tmp / "best.pth", link)
    shutil.copy2(run_dir / "best.json", tmp / "best.json")
    snapshot_note = ".hydra/config.yaml (training config snapshot)"
    if hydra_dir is not None:
        (tmp / ".hydra").mkdir()
        for name in SNAPSHOT_FILES:
            if (hydra_dir / ".hydra" / name).exists():
                shutil.copy2(hydra_dir / ".hydra" / name, tmp / ".hydra" / name)
    else:
        cfg, source = _vw().resolve_config(row, master_rows or [])
        if cfg is not None:
            import yaml
            (tmp / ".hydra").mkdir()
            (tmp / ".hydra" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            (tmp / ".hydra" / "SNAPSHOT_SOURCE.txt").write_text(
                f"{source}\n# hydra 스냅샷이 없는 run — 위 출처(다른 seed 의 스냅샷 또는 학습 로그)에서 재구성한 설정. seed·run.name 외 설정은 시리즈 안에서 같다.\n")
            snapshot_note = f".hydra/config.yaml (reconstructed from {source}; see .hydra/SNAPSHOT_SOURCE.txt)"
        else:
            snapshot_note = "no .hydra (not a hydra training run: MVST fusion MLP — evaluate with scripts/verify_paper_weights.py together with the five view bundles)"
    best = json.loads((tmp / "best.json").read_text())
    (tmp / "README.txt").write_text(
        f"RSC-Toolkit paper weight: {run}\n"
        f"family={family_of(run)} dataset={dataset_of(row)} model={row['model_name']} config={row['config_name']}\n"
        f"best.json: S_p={best.get('S_p'):.2f} S_e={best.get('S_e'):.2f} Score={best.get('icbhi_score'):.2f} (epoch {best.get('epoch')})\n\n"
        f"Files: best.pth (model state_dict under key 'model'), best.json (metrics at the saved epoch), {snapshot_note}.\n"
        "Re-evaluate (from a git clone of RSC-Toolkit, dataset downloaded and indexed):\n"
        f"  rsc-toolkit-eval <this folder> --project-root <clone> --override data.params.data_dir=<dataset dir>\n"
        "See TUTORIAL.md.\n")
    files = sorted(p for p in tmp.rglob("*") if p.is_file() and p.name != "sha256.txt")
    lines = [f"{sha256(p)}  {p.relative_to(tmp).as_posix()}" for p in files]
    (tmp / "sha256.txt").write_text("\n".join(lines) + "\n")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return {"run": run, "family": family_of(run), "dataset": dataset_of(row), "model": row["model_name"], "config_name": row["config_name"],
            "S_p": f"{best.get('S_p'):.4f}", "S_e": f"{best.get('S_e'):.4f}", "icbhi_score": f"{best.get('icbhi_score'):.4f}", "epoch": best.get("epoch"),
            "best_pth_sha256": sha256(dst / "best.pth"), "best_pth_bytes": (dst / "best.pth").stat().st_size,
            "bundle_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(), "url": URL_PLACEHOLDER}


def write_manifest(rows: dict) -> None:
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for run in sorted(rows):
            w.writerow({k: rows[run].get(k, "") for k in FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--family", action="append", default=[], help="family 이름 → seed 1–5 run 전부")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "weights_release"))
    ap.add_argument("--all-paper", action="store_true", help="docs/paper/anchors.csv 의 표 1·2 family 전부(+MVST 뷰 family)")
    ap.add_argument("--include-m2d", action="store_true", help="--all-paper 에서 M2D 계열(NTT 평가 전용 라이선스)도 포함")
    ap.add_argument("--link", action="store_true", help="best.pth 를 복사 대신 하드링크 (같은 파일시스템; 추가 용량 0)")
    ap.add_argument("--layout", choices=["flat", "nested"], default="flat", help="nested = <out>/<dataset>/<family>/<run>/")
    args = ap.parse_args()
    master = {r["run_name"]: r for r in csv.DictReader(open(MASTER))}
    families = list(args.family)
    if args.all_paper:
        families += paper_families(include_m2d=args.include_m2d)
    runs = list(dict.fromkeys(list(args.run) + [r for f in families for r in master if family_of(r) == f and (re.search(r"-s[1-5]$", r) or r == f)]))
    if not runs:
        ap.error("--run / --family / --all-paper 중 하나를 주어라")
    for run in runs:                                            # 전부 먼저 검사 — 하나라도 못 만들면 아무것도 건드리지 않는다
        if run not in master:
            raise SystemExit(f"{run} 이 results_master 에 없다")
        check_row(master[run])
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = {r["run"]: r for r in csv.DictReader(open(MANIFEST))} if MANIFEST.exists() else {}
    master_rows = list(master.values())
    made: list[tuple[str, pathlib.Path, dict]] = []
    for run in runs:
        row = master[run]
        run_out = out / dataset_of(row) / family_of(run) if args.layout == "nested" else out
        rec = bundle(row, run_out, master_rows, link=args.link)
        if run in rows and rows[run].get("url") not in ("", URL_PLACEHOLDER):
            rec["url"] = rows[run]["url"]          # 이미 올린 링크는 보존
        rows[run] = rec
        write_manifest(rows)                       # run 마다 갱신 — 뒤의 run 이 실패해도 대장과 디스크가 어긋나지 않는다
        made.append((run, run_out / run, rec))
        print(f"{run:28s} → {run_out / run}  best.pth {rec['best_pth_bytes']/1e6:.1f} MB  sha256 {rec['best_pth_sha256'][:12]}…", flush=True)
    write_index(out, made)
    print(f"manifest: {MANIFEST} ({len(rows)} rows); index: {out / 'INDEX.tsv'}")
    return 0


def paper_families(include_m2d: bool = False) -> list[str]:
    """표 1·2 의 family (anchors.csv, pm_order 있는 A/B 행, expect_n>0). MVST 융합 family 에는 뷰 family 5개를 붙인다."""
    anch = [{k.strip(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(open(ROOT / "docs" / "paper" / "anchors.csv", newline=""))]
    meta = {r["family"]: r for r in csv.DictReader(open(ROOT / "docs" / "paper" / "model_meta.csv"))}
    fams: list[str] = []
    for a in anch:
        if not a["pm_order"] or a["table"] not in ("A", "B") or a["expect_n"] in ("", "0"):
            continue
        f = a["family"]
        if f in fams:
            continue
        if not include_m2d and meta.get(f, {}).get("arch", "") == "M2D":
            continue
        fams.append(f)
        if f == "icbhi-mvst-official-fusion":
            fams += [f"icbhi-mvst-official-v{v}" for v in (16, 32, 64, 128, 256)]
        elif f == "sprs-mvst-fusion":
            fams += [f"sprs-mvst-v{v}" for v in (16, 32, 64, 128, 256)]
    return fams


def write_index(out: pathlib.Path, made: list[tuple[str, pathlib.Path, dict]]) -> None:
    """<out>/INDEX.tsv (run 단위) + <out>/README.md (family 단위 크기·업로드 안내). 기존 항목은 유지하고 갱신한다."""
    idx = out / "INDEX.tsv"
    rows: dict[str, dict] = {}
    if idx.exists():
        for r in csv.DictReader(open(idx), delimiter="\t"):
            rows[r["run"]] = r
    for run, path, rec in made:
        rows[run] = {"run": run, "family": rec["family"], "dataset": rec["dataset"], "model": rec["model"], "path": str(path.relative_to(out)),
                     "best_pth_MB": f"{rec['best_pth_bytes']/1e6:.1f}", "score": rec["icbhi_score"], "best_pth_sha256": rec["best_pth_sha256"]}
    with open(idx, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "family", "dataset", "model", "path", "best_pth_MB", "score", "best_pth_sha256"], delimiter="\t")
        w.writeheader()
        for run in sorted(rows):
            w.writerow(rows[run])
    fam: dict[tuple[str, str], list[dict]] = {}
    for r in rows.values():
        fam.setdefault((r["dataset"], r["family"]), []).append(r)
    total = sum(float(r["best_pth_MB"]) for r in rows.values())
    L = ["# RSC-Toolkit paper weights — release bundles", "",
         f"{len(rows)} run bundles · {len(fam)} families · {total/1e3:.1f} GB of best.pth. One folder per run: best.pth, best.json, .hydra/config.yaml, README.txt, sha256.txt.",
         "Layout: `<dataset>/<family>/<run>/`. Upload the whole tree (keep folder names; include the hidden `.hydra/` folders).",
         "Index of every run: `INDEX.tsv`.", "",
         "| dataset | family | runs | best.pth total |", "|---|---|---|---|"]
    for (ds, f) in sorted(fam):
        rs = fam[(ds, f)]
        L.append(f"| {ds} | `{f}` | {len(rs)} | {sum(float(r['best_pth_MB']) for r in rs)/1e3:.2f} GB |")
    L += ["", "Re-evaluate a bundle: `rsc-toolkit-eval <run folder> --project-root <clone> --override data.params.data_dir=<dataset dir>` (TUTORIAL.md).",
          "MVST: the `*-fusion-s<k>` folder holds the fusion MLP; the five `*-v{16,32,64,128,256}-s<k>` view bundles of the same seed are required (scripts/verify_paper_weights.py)."]
    (out / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
