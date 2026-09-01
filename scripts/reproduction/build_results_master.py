# -*- coding: utf-8 -*-
"""outputs/ 전체를 훑어 논문 표의 **단일 진실원** `results_master.csv` 를 만든다.

수치를 손으로 옮기지 않기 위한 경로다. §3-D QLung 칸이 "54.68 (1/5, 진행 중)" 으로 굳은 채
실제로는 5/5 완주(56.66±3.67)였던 것이 손 이관의 대가였다.

조인 방식 — `best.json`(런 결과) 과 `.hydra` 스냅샷(런 설정)을 `run.name` 으로 붙인다.
성립 근거: hydra 가 `config.yaml` 에 `run.name`/`run.group` 을 그대로 남기고,
그것이 `outputs/<group>/<run.name>/` 디렉터리명과 같다(전수 확인).

같은 `run.name` 으로 여러 스냅샷이 있으면(재실행) **경로 사전순 마지막**을 쓴다 —
스냅샷 디렉터리가 `outputs/YYYY-MM-DD/HH-MM-SS/` 라 사전순 = 시간순이고,
살아남은 `best.json` 은 마지막 실행의 것이기 때문이다.

사용:
    python scripts/reproduction/build_results_master.py                 # docs/results_master.csv
    python scripts/reproduction/build_results_master.py --out /tmp/x.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# best.json 안에서 그대로 실어 나르는 지표 키. 툴킷 표준 3종이 먼저다.
STD_KEYS = ["S_e", "S_p", "icbhi_score", "epoch"]
# 비표준 선택 규칙이 남긴 흔적 — AMS-D 이진 지표(paper_*) 와 최종 iterate(final_*).
# 어느 규칙이 고른 값인지 표에서 구분해야 해서 같이 싣는다.
ALT_KEYS = [
    "selection_key",
    "paper_S_e", "paper_S_p", "paper_score", "paper_acc",
    "final_S_e", "final_S_p", "final_icbhi_score", "final_epoch",
]


def iter_run_dirs():
    """결과가 있는 런 디렉터리를 모두 낸다. 날짜 디렉터리(= hydra 작업 디렉터리)는 건너뛴다."""
    for best in sorted(OUTPUTS.glob("**/best.json")):
        rel = best.relative_to(OUTPUTS)
        if DATE_DIR.match(rel.parts[0]):
            continue
        yield best.parent, rel


def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # 깨진 파일은 조용히 넘기지 않는다
        print(f"  ! 읽기 실패 {path}: {exc}", file=sys.stderr)
        return None


def index_hydra_snapshots():
    """run.name -> {config, hydra, dir} 색인. 중복은 경로 사전순 마지막이 이긴다."""
    index, dupes = {}, {}
    for cfg_path in sorted(OUTPUTS.glob("*/*/.hydra/config.yaml")):
        if not DATE_DIR.match(cfg_path.relative_to(OUTPUTS).parts[0]):
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception as exc:
            print(f"  ! config 파싱 실패 {cfg_path}: {exc}", file=sys.stderr)
            continue
        name = ((cfg or {}).get("run") or {}).get("name")
        if not name:
            continue
        hyd_path = cfg_path.parent / "hydra.yaml"
        hyd = {}
        if hyd_path.exists():
            try:
                hyd = yaml.safe_load(hyd_path.read_text()) or {}
            except Exception:
                hyd = {}
        ovr_path = cfg_path.parent / "overrides.yaml"
        overrides = []
        if ovr_path.exists():
            try:
                overrides = yaml.safe_load(ovr_path.read_text()) or []
            except Exception:
                overrides = []
        dupes[name] = dupes.get(name, 0) + 1
        index[name] = {"cfg": cfg, "hydra": hyd, "dir": cfg_path.parent.parent, "overrides": overrides}
    return index, {k: v for k, v in dupes.items() if v > 1}


def index_done_markers():
    """완주 마커 색인. 마커는 결과 디렉터리가 아니라 **런처 번들** 쪽에 쌓인다
    (`outputs/<bundle>/logs/<run>.done`) — 같은 런의 결과는 `outputs/<계열>/<run>/` 에 있어
    디렉터리가 서로 다르다. 그래서 이름으로만 붙일 수 있다."""
    marks = {}
    for mk in OUTPUTS.glob("*/logs/*.done"):
        try:
            body = mk.read_text()
        except Exception:
            continue
        marks[mk.stem] = "1" if "EXIT=0" in body else "0"
    return marks


def load_hparams(run_name: str) -> dict:
    """hydra 스냅샷이 없는 초기 런은 lightning_logs/<run>/version_*/hparams.yaml 로 일부 복구된다."""
    base = ROOT / "lightning_logs" / run_name
    for hp in sorted(base.glob("version_*/hparams.yaml")):
        try:
            return yaml.safe_load(hp.read_text()) or {}
        except Exception:
            continue
    return {}


def dig(d, *path, default=""):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def row_for(run_dir: pathlib.Path, rel: pathlib.Path, snap: dict | None, marks: dict) -> dict:
    best = load_json(run_dir / "best.json") or {}
    parts = rel.parts
    group = parts[0]
    run_name = run_dir.name
    # outputs/<group>/<parent>/<sub>/best.json 처럼 한 겹 더 들어간 변형 런이 3건 있다.
    parent_run = parts[1] if len(parts) > 3 else ""

    row = {
        "paper_group": group,
        "run_name": run_name,
        "parent_run": parent_run,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    for key in STD_KEYS + ALT_KEYS:
        row[key.lower() if key.startswith("S_") else key] = best.get(key, "")

    ood = run_dir / "ood.json"
    row["has_ood"] = "1" if ood.exists() else ""
    row["provenance"] = "none"
    row["derived_from"] = ""
    row["done"] = marks.get(run_name, "")

    if not snap:
        # 폴백 1 — best.json 이 스스로 출처를 담는 경우(예: mvst_fusion.py 가 split·ckpts 를 적는다)
        if best.get("split") or best.get("ckpts"):
            row["provenance"] = "self"
            row["split_file"] = best.get("split", "")
            row["n_cls"] = best.get("n_cls", "")
            row["dataset"] = best.get("dataset", "")
            row["derived_from"] = "|".join(best.get("ckpts") or [])
            return row
        # 폴백 2 — lightning_logs 의 hparams.yaml (모델·데이터 이름까지만 복구된다)
        hp = load_hparams(run_name)
        if hp:
            row["provenance"] = "hparams"
            row["model_name"] = hp.get("model/name", "")
            row["config_name"] = hp.get("data/name", "")
            row["sample_level"] = hp.get("data/sample_level", "")
            return row
        row["provenance"] = "none"
        return row

    cfg, hyd = snap["cfg"], snap["hydra"]
    params = dig(cfg, "data", "params", default={}) or {}
    row.update({
        "provenance": "hydra",
        "config_name": dig(hyd, "hydra", "job", "config_name"),
        "seed": cfg.get("seed", ""),
        "dataset": params.get("dataset_name", ""),
        "sample_level": params.get("sample_level", ""),
        "split_file": params.get("split_file", ""),
        "split_mode": params.get("split_mode", ""),
        "train_splits": "|".join(params.get("train_splits") or []),
        "val_splits": "|".join(params.get("val_splits") or []),
        "test_splits": "|".join(params.get("test_splits") or []),
        "model_name": dig(cfg, "model", "name"),
        "n_cls": dig(cfg, "model", "params", "n_cls"),
        "loss_name": dig(cfg, "loss", "name"),
        "max_epochs": dig(cfg, "trainer", "params", "max_epochs"),
        "lr": dig(cfg, "optimizer", "params", "lr"),
        "overrides": ",".join(str(o) for o in snap["overrides"]),
        "hydra_dir": str(snap["dir"].relative_to(ROOT)),
    })
    return row


COLUMNS = [
    # 신원
    "paper_group", "run_name", "parent_run", "seed",
    # 툴킷 표준 지표 3종 — 어떤 표에서도 이 셋을 다 채운다
    "s_e", "s_p", "icbhi_score", "epoch",
    # 설정·프로토콜
    "dataset", "config_name", "model_name", "n_cls", "loss_name",
    "split_file", "split_mode", "train_splits", "val_splits", "test_splits",
    "sample_level", "max_epochs", "lr",
    # 비표준 선택 규칙의 흔적
    "selection_key", "paper_S_e", "paper_S_p", "paper_score", "paper_acc",
    "final_S_e", "final_S_p", "final_icbhi_score", "final_epoch",
    # 출처
    "done", "has_ood", "provenance", "derived_from", "overrides", "run_dir", "hydra_dir",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results_master.csv"))
    args = ap.parse_args()

    index, dupes = index_hydra_snapshots()
    marks = index_done_markers()
    print(f"hydra 스냅샷 {len(index)} 개 (중복 run.name {len(dupes)} 건 — 마지막 실행 채택)", file=sys.stderr)

    rows, unmatched = [], []
    for run_dir, rel in iter_run_dirs():
        snap = index.get(run_dir.name)
        if snap is None:
            unmatched.append(str(run_dir.relative_to(ROOT)))
        rows.append(row_for(run_dir, rel, snap, marks))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})

    # 지표 3종이 다 있는지는 논문 규약(D7)의 전제라 매번 확인한다.
    missing = [r["run_name"] for r in rows if any(r.get(k) in ("", None) for k in ("s_e", "s_p", "icbhi_score"))]
    print(f"런 {len(rows)} 개 → {out.relative_to(ROOT)}", file=sys.stderr)
    prov = collections.Counter(r.get("provenance") or "none" for r in rows)
    print(f"  출처: " + " · ".join(f"{k} {v}" for k, v in prov.most_common()), file=sys.stderr)
    print(f"  Se/Sp/Score 결측 {len(missing)}", file=sys.stderr)
    dn = collections.Counter(r.get("done") or "마커없음" for r in rows)
    print(f"  완주 마커: " + " · ".join(f"{'EXIT=0' if k=='1' else ('EXIT≠0' if k=='0' else k)} {v}" for k, v in dn.most_common()), file=sys.stderr)
    orphans = [r["run_name"] for r in rows if (r.get("provenance") or "none") == "none"]
    if orphans:
        print(f"  설정 복구 불가 {len(orphans)} 건 — 논문 아티팩트 진술에서 이 런들은 별도 표기해야 한다:", file=sys.stderr)
        for name in orphans:
            print(f"    - {name}", file=sys.stderr)
    if missing:
        print(f"  결측 런: {missing[:20]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
