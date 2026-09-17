#!/usr/bin/env python
"""MVST (fusion) 의 ICBHI→SPRSound 제로샷 — 뷰 5개 특징 → ICBHI 로 학습한 FuseMLP (2026-09-15 사용자 결정).

scripts/eval_ood_sprsound.py 는 run 하나 = 모델 하나를 전제하므로 MVST 융합(뷰 best.pth 5개 + FuseMLP best.pth)은 이 스크립트가 맡는다.
특징 경로는 융합을 학습할 때(scripts/reproduction/mvst_fusion.py `dump`)와 같다:
  뷰 모델(학습 설정 스냅샷으로 재구성, best.pth strict 로드).eval() → forward_features(to_model_input(x)) (autocast) → 768-d
데이터는 eval_ood_sprsound.make_ood_data_cfg 의 일반 분기(뷰 run 의 ICBHI fbank 체인을 SPRSound 2022 inter-patient test 에 적용, 7→4 병합 인덱스).
융합: FuseMLP(num_views=5, num_classes=4) 에 융합 run 의 best.pth 를 strict 로드, argmax → 4-class S_p/S_e/Score (get_score, ICBHI 산식).

검산(--dataset icbhi): 같은 코드를 뷰 run 의 원 설정(ICBHI 공식 test)에 돌려 융합 run 의 best.json 과 대조한다 — 이것이 맞아야 특징 경로가 학습 때와 같다는 뜻.

실행:
  .venv/bin/python scripts/eval_ood_mvst_fusion.py --out DIR --dataset icbhi   --seeds 1,2,3,4,5 --no-write-run   # 검산
  .venv/bin/python scripts/eval_ood_mvst_fusion.py --out DIR --dataset sprsound --seeds 1,2,3,4,5                  # 제로샷 → ood.json
출력: DIR/results.jsonl, outputs/mvst/<fusion run>/ood.json ({"inter": {...}}; 없을 때만 씀)
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("eval_ood_sprsound", ROOT / "scripts" / "eval_ood_sprsound.py")
eo = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eo)  # type: ignore[union-attr]
vw = eo.vw

import torch  # noqa: E402

VIEWS = [16, 32, 64, 128, 256]
FAMILY = "icbhi-mvst-official-fusion"


def load_view(view_row: dict, master: list[dict], sandbox: pathlib.Path, device: str):
    from rsc_toolkit.trainer.factories.build_model import build_model
    cfg, src = vw.resolve_config(view_row, master)
    if cfg is None:
        raise RuntimeError(f"{view_row['run_name']}: 설정 스냅샷 없음")
    cfg, _ = vw.sanitize_config(cfg, sandbox, view_row["run_name"])
    model = build_model(copy.deepcopy(cfg["model"]))
    sd = torch.load(ROOT / view_row["run_dir"] / "best.pth", map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    vw._adapt_buffer_shapes(model, sd)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), cfg, src


def build_loader(cfg: dict, dataset: str):
    """뷰 run 의 data cfg → (loader, chain note, 4-class counts|None). ICBHI 는 원 설정의 val(=공식 test), SPRSound 는 OOD 뷰."""
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
    if dataset == "sprsound":
        data_cfg, note = eo.make_ood_data_cfg(cfg["data"], FAMILY)
    else:
        data_cfg, note = copy.deepcopy(cfg["data"]), "ICBHI official 60/40 test (sanity check against fusion best.json)"
    dm = build_datamodule(copy.deepcopy(data_cfg))
    dm.setup(None)
    counts = eo.remap_labels(dm) if dataset == "sprsound" else None
    return dm.val_dataloader(), note, counts


@torch.no_grad()
def dump_features(models: dict, loader, device: str):
    """scripts/reproduction/mvst_fusion.py `dump` 와 같은 경로 (x.squeeze(1) → to_model_input → forward_features, autocast)."""
    amp = device.startswith("cuda")
    feats = {v: [] for v in models}; labels = []
    for batch in loader:
        x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch["x"], batch["y"])
        x = x.squeeze(1).to(device)
        with torch.autocast(device_type="cuda", enabled=amp):
            for v, m in models.items():
                feats[v].append(m.forward_features(m.to_model_input(x)).float().cpu())
        labels.append(y.cpu())
    return [torch.cat(feats[v]) for v in VIEWS], torch.cat(labels)


@torch.no_grad()
def fuse_predict(fusion_dir: pathlib.Path, feats, labels, device: str, n_cls: int):
    """verify_paper_weights.verify_fusion 과 같은 적용 경로(FuseMLP strict 로드, bs 8, autocast)."""
    from rsc_toolkit.methods.mvst_fusion import FuseMLP, get_score
    fuse = FuseMLP(num_views=len(feats), num_classes=n_cls)
    sd = torch.load(fusion_dir / "best.pth", map_location="cpu", weights_only=False)["model"]
    fuse.load_state_dict(sd, strict=True)
    fuse = fuse.to(device).eval()
    amp = device.startswith("cuda")
    hits, counts = [0.0] * n_cls, [0.0] * n_cls
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*feats, labels), batch_size=8, shuffle=False, num_workers=0)
    for *fs, y in loader:
        fs = [f.to(device) for f in fs]
        with torch.autocast(device_type="cuda", enabled=amp):
            pred = fuse(*fs).argmax(1)
        for p, t in zip(pred.tolist(), y.tolist()):
            counts[t] += 1
            if p == t:
                hits[t] += 1
    sp, se, sc = get_score(hits, counts)
    return {"S_p": sp, "S_e": se, "icbhi_score": sc}, [int(c) for c in counts], [int(h) for h in hits]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="sprsound", choices=["sprsound", "icbhi"])
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--family", default=FAMILY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-write-run", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05, help="ICBHI 검산 허용 |Δ| (점)")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "sprsound":
        eo.ensure_merged_index()
    sandbox = out / "sandbox"; vw.make_sandbox(sandbox)
    master = list(csv.DictReader(open(ROOT / "docs" / "results_master.csv")))
    by_run = {r["run_name"]: r for r in master}
    by_dir = {r["run_dir"]: r for r in master}
    log = open(out / "results.jsonl", "a")
    for seed in (int(s) for s in args.seeds.split(",")):
        run = f"{args.family}-s{seed}"
        rec = {"family": args.family, "run": run, "dataset": args.dataset, "status": "ERROR", "ts": time.time()}
        t0 = time.time()
        try:
            row = by_run.get(run)
            if row is None:
                raise KeyError(f"{run} 이 results_master 에 없다")
            run_dir = ROOT / row["run_dir"]
            if args.dataset == "sprsound" and (run_dir / "ood.json").exists() and not args.no_write_run:
                rec.update(status="SKIP", note="ood.json 이미 있음"); continue
            best = json.loads((run_dir / "best.json").read_text())
            ckpts = best["ckpts"]; n_cls = int(best.get("n_cls") or 4)
            import lightning as L
            L.seed_everything(seed, workers=True)
            models, sources, view_cfg = {}, {}, None
            for v in VIEWS:
                vrow = by_dir.get(str(pathlib.Path(ckpts[str(v)]).parent))
                if vrow is None:
                    raise KeyError(f"뷰 run 을 results_master 에서 못 찾음: {ckpts[str(v)]}")
                models[v], cfg, sources[v] = load_view(vrow, master, sandbox, args.device)
                view_cfg = view_cfg or cfg
            loader, note, counts4 = build_loader(view_cfg, args.dataset)
            feats, labels = dump_features(models, loader, args.device)
            del models; torch.cuda.empty_cache()
            m, counts, hits = fuse_predict(run_dir, feats, labels, args.device, n_cls)
            rec.update(metrics=m, class_counts=counts, hits=hits, n=int(labels.numel()), chain=note, config_sources=sources,
                       views={str(v): ckpts[str(v)] for v in VIEWS})
            if args.dataset == "icbhi":
                diffs = {k: round(m[k] - float(best[k]), 4) for k in ("S_p", "S_e", "icbhi_score")}
                rec["expected"] = {k: best[k] for k in ("S_p", "S_e", "icbhi_score")}
                rec["diffs"] = diffs
                rec["status"] = "PASS" if max(abs(d) for d in diffs.values()) <= args.tol else "FAIL"
            else:
                if counts4 != eo.EXPECTED_COUNTS or counts != eo.EXPECTED_COUNTS:
                    rec["status"] = "COUNT_MISMATCH"
                else:
                    rec["status"] = "OK"
                    if not args.no_write_run:
                        payload = {"inter": {**m, "n": 1429, "class_counts": counts},
                                   "protocol": "ICBHI-trained MVST: five view best.pth → SPRSound 2022 inter-patient test features "
                                               "(recipe's ICBHI fbank chain applied to SPRSound), ICBHI-trained FuseMLP best.pth applied; 7→4 merged (rhonchi/stridor excluded; none in the test set)",
                                   "chain": note, "views": {str(v): ckpts[str(v)] for v in VIEWS},
                                   "generated_by": "scripts/eval_ood_mvst_fusion.py", "generated": time.strftime("%Y-%m-%d %H:%M")}
                        (run_dir / "ood.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
            print(f"[{run}/{args.dataset}] {rec['status']} Sp {m['S_p']:.2f} Se {m['S_e']:.2f} Sc {m['icbhi_score']:.2f} "
                  f"n={rec['n']} counts={counts} {rec.get('diffs', '')} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 — seed 하나의 실패가 전체를 멈추지 않게
            rec["error"] = f"{type(exc).__name__}: {exc}"[:800]; rec["traceback"] = traceback.format_exc()[-2000:]
            print(f"[{run}/{args.dataset}] ERROR {rec['error']}", flush=True)
        finally:
            rec["seconds"] = round(time.time() - t0, 1)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
