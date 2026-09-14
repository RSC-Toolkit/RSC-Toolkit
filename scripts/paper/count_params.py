#!/usr/bin/env python
"""표 2 의 #Param (M) 열 원천 `docs/paper/params.csv` 를 만든다.

방법 — 표 2(B: SPRSound→SPRSound, C: ICBHI→SPRSound) 의 각 family 에 대해 대표 run(이름순 첫 run, best.pth 있는 것)의
설정 스냅샷으로 **학습 때와 같은 팩토리**(`build_model`)로 모델을 세우고 `parameters()` 를 센다(버퍼 제외; 공유 파라미터는 1회).
  * params_total     : 추론 모델 전체(동결 백본 포함) — 표의 #Param
  * params_trainable : 생성 직후 requires_grad=True 인 것(PEFT·동결 백본 행은 total 과 다르다; 동결은 모델 __init__ 에서 일어난다)
  * 교차 확인        : best.pth 의 텐서 원소 합 == total + buffers 이어야 한다(strict 로드가 성립하므로). tied weight 가 있으면 ckpt 쪽이 크다.
  * MVST (fusion)    : best.json 의 뷰 ckpt 5개(각 뷰 모델) + FuseMLP 의 합.
SPRSound 쪽(B)과 ICBHI 쪽(C)은 분류 헤드 크기(7 vs 4 class)만 달라 0.1 M 해상도에서는 같다 — 둘 다 세어 차이를 기록한다.

실행: .venv/bin/python scripts/paper/count_params.py [--out docs/paper/params.csv] [--family F ...]
(GPU 불필요, outputs/ 는 읽기만; 샌드박스는 scratch 디렉터리에 만든다.)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import re
import sys
import tempfile
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "paper"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

import build_paper_tables as bpt  # noqa: E402
import verify_paper_weights as vw  # noqa: E402
from rsc_toolkit import evaluate as ev  # noqa: E402

COLUMNS = ["family", "side", "run_name", "config_source", "model_name", "params_total", "params_trainable", "buffers",
           "ckpt_tensor_numel", "ckpt_keys", "check", "note"]


def count_module(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return total, trainable, buffers


def count_state_dict(path: pathlib.Path) -> tuple[int, int]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    return sum(int(v.numel()) for v in sd.values() if torch.is_tensor(v)), len(sd)


def pick_run(fam_rows: list[dict]) -> dict | None:
    for r in sorted(fam_rows, key=lambda r: r["run_name"]):
        if (ROOT / r["run_dir"] / "best.pth").exists():
            return r
    return None


def build_from_row(row: dict, master_rows: list[dict], sandbox: pathlib.Path) -> tuple[torch.nn.Module, str, str]:
    cfg, src = vw.resolve_config(row, master_rows)
    if cfg is None:
        raise RuntimeError("설정 스냅샷·로그 없음")
    cfg, _ = vw.sanitize_config(cfg, sandbox, row["run_name"])
    from rsc_toolkit.trainer.factories.build_model import build_model
    with ev._chdir(sandbox):
        model = build_model(copy.deepcopy(cfg["model"]))
    return model, src, str(cfg["model"].get("name", ""))


def count_fusion(row: dict, master_by_dir: dict, master_rows: list[dict], sandbox: pathlib.Path) -> dict:
    """MVST 융합: 뷰 ckpt 5개의 모델 + FuseMLP. best.json 의 ckpts 경로 → results_master 의 run 행 → 같은 팩토리로 생성."""
    from rsc_toolkit.methods.mvst_fusion import FuseMLP
    best = json.loads((ROOT / row["run_dir"] / "best.json").read_text())
    ckpts = best.get("ckpts") or {}
    total = trainable = buffers = ck = keys = 0
    parts = []
    for view, rel in ckpts.items():
        vrow = master_by_dir.get(str(pathlib.Path(rel).parent))
        if vrow is None:
            raise RuntimeError(f"뷰 run 을 results_master 에서 못 찾음: {rel}")
        m, src, name = build_from_row(vrow, master_rows, sandbox)
        t, tr, b = count_module(m)
        c, k = count_state_dict(ROOT / rel)
        total += t; trainable += tr; buffers += b; ck += c; keys += k
        parts.append(f"v{view}:{name}:{t/1e6:.2f}M")
        del m
    n_cls = int(best.get("n_cls") or 0)
    fuse = FuseMLP(num_views=len(ckpts), num_classes=n_cls)
    t, tr, b = count_module(fuse)
    c, k = count_state_dict(ROOT / row["run_dir"] / "best.pth")
    total += t; trainable += tr; buffers += b; ck += c; keys += k
    parts.append(f"fuse:{t/1e6:.3f}M")
    return {"model_name": "mvst_fusion(5 views + FuseMLP)", "params_total": total, "params_trainable": trainable, "buffers": buffers,
            "ckpt_tensor_numel": ck, "ckpt_keys": keys, "config_source": "self+views", "note": " + ".join(parts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "paper" / "params.csv"))
    ap.add_argument("--family", nargs="*", default=None, help="이 family 만 (기본: 표 2 의 B·C 전부)")
    ap.add_argument("--resume", action="store_true", help="기존 --out 에 있는 (family, side) 는 건너뛴다 (행마다 즉시 저장하므로 중단 후 이어하기)")
    args = ap.parse_args()

    anchors = list(csv.DictReader(open(bpt.ANCHORS if hasattr(bpt, "ANCHORS") else ROOT / "docs" / "paper" / "anchors.csv")))
    master_rows, fam = bpt.load_master()
    master_by_dir = {r["run_dir"]: r for r in master_rows}
    targets = [(a["family"], a["table"]) for a in anchors if a["table"] in "BC" and a.get("pm_order")]
    if args.family:
        targets = [t for t in targets if t[0] in set(args.family)]

    sandbox = pathlib.Path(tempfile.mkdtemp(prefix="count_params_"))
    vw.make_sandbox(sandbox)
    out = pathlib.Path(args.out)
    out_rows = []
    if args.resume and out.exists():
        out_rows = [r for r in csv.DictReader(open(out)) if r["check"] not in ("error", "")]
        done = {(r["family"], r["side"]) for r in out_rows}
        targets = [t for t in targets if t not in done]
        print(f"resume: 기존 {len(out_rows)} 행, 남은 {len(targets)}", flush=True)

    def flush():   # 행마다 저장 — 모델 생성 중 프로세스가 죽어도(네이티브 확장 등) 앞선 결과는 남는다
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader(); w.writerows(out_rows)
    t0 = time.time()
    for family, side in targets:
        rec = {c: "" for c in COLUMNS}
        rec.update(family=family, side=side)
        row = pick_run(fam.get(family, []))
        if row is None:
            rec["check"] = "no-run"; rec["note"] = "best.pth 있는 run 없음"
            out_rows.append(rec); flush(); print(f"[{family}] no run", flush=True); continue
        print(f"[{family}/{side}] {row['run_name']} …", flush=True)
        rec["run_name"] = row["run_name"]
        try:
            if row.get("provenance") == "self" and "fusion" in family:
                rec.update(count_fusion(row, master_by_dir, master_rows, sandbox))
            else:
                model, src, name = build_from_row(row, master_rows, sandbox)
                total, trainable, buffers = count_module(model)
                ck, keys = count_state_dict(ROOT / row["run_dir"] / "best.pth")
                rec.update(config_source=src, model_name=name, params_total=total, params_trainable=trainable, buffers=buffers,
                           ckpt_tensor_numel=ck, ckpt_keys=keys)
                del model
            diff = int(rec["ckpt_tensor_numel"]) - (int(rec["params_total"]) + int(rec["buffers"]))
            rec["check"] = "ok" if diff == 0 else f"ckpt-(params+buffers)={diff:+d}"
            print(f"[{family}/{side}] {rec['run_name']} {rec['model_name']}: total {int(rec['params_total'])/1e6:.2f}M "
                  f"trainable {int(rec['params_trainable'])/1e6:.2f}M buffers {int(rec['buffers'])/1e6:.3f}M {rec['check']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 — 한 family 실패가 전체를 막지 않게
            rec["check"] = "error"; rec["note"] = f"{type(exc).__name__}: {exc}"
            print(f"[{family}/{side}] ERROR {exc}\n{traceback.format_exc()}", flush=True)
        out_rows.append(rec); flush()

    out_rows.sort(key=lambda r: (r["side"], r["family"])); flush()
    bad = [r["family"] for r in out_rows if r["check"] not in ("ok",) and not r["check"].startswith("ckpt-")]
    print(f"→ {out.relative_to(ROOT)} · {len(out_rows)} 행 · 실패/결측 {len(bad)}: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
