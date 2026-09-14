# -*- coding: utf-8 -*-
"""
ICBHI 학습 체크포인트 → SPRSound 2022 inter-patient test 제로샷 OOD 평가 (Table 4, 2026-09-08 사용자 결정).

원칙: 각 run 의 설정 스냅샷으로 **학습 때와 같은 팩토리**로 모델·데이터 체인을 재구성하고(best.pth strict 로드),
데이터셋만 SPRSound 로 바꾼다 — 그 recipe 의 ICBHI 전처리 체인을 SPRSound 에 그대로 적용, 라벨은 7→4 병합
(Normal→0, Fine/Coarse Crackle→1, Wheeze/Rhonchi/Stridor→2, Both→3; inter-test 는 1040/83/305/1). 지표 = ICBHI 4-class Sp/Se/Score.
QLung·BTS-CARD 의 기존 ood.json(같은 정의) 은 건드리지 않는다.

  .venv/bin/python scripts/eval_ood_sprsound.py --out DIR [--families f1,f2] [--seeds 1,2,3,4,5] [--device cuda] [--no-write-run]

출력: DIR/results.jsonl (run 단위 기록), DIR/summary.md, 그리고 outputs/<group>/<run>/ood.json (없을 때만; {"inter": {...}}).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("verify_paper_weights", ROOT / "scripts" / "verify_paper_weights.py")
vw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(vw)  # type: ignore[union-attr]

import torch  # noqa: E402

# 파생 인덱스: index_cycle.csv 의 label 을 7→4 로 병합해 둔 사본(.datasets/sprsound/index/index_cycle_icbhi4.csv).
# 데이터모듈이 생성 시점에 라벨 수를 세는 경우(PC-MCL·Prototype·Domain Transfer)까지 한 경로로 처리하기 위해 CSV 단계에서 병합한다.
SPRS = {"dataset_name": "sprsound", "data_dir": str(ROOT / ".datasets" / "sprsound"), "raw_data_subdir": "Classification",
        "index_filename": "index_cycle_icbhi4.csv"}
EXPECTED_COUNTS = [1040, 83, 305, 1]
CACHE_ROOT = ROOT / "downloaded_datasets" / "features_cache_ood"
# 기본 대상: 표 1 의 ICBHI family 중 OOD 가 정의되는 것 (QLung·BTS-CARD 는 기존 ood.json, RespireNet·GaP-Aug·DDE-MAE·MVST 제외)
DEFAULT_FAMILIES = [
    "icbhi-ast-sft", "icbhi-patchmix-ce", "icbhi-patchmix-cl", "icbhi-ast-sgscl-githead", "icbhi-lungadapter", "icbhi-addrsc",
    "icbhi-lungmix-nonlinear", "icbhi-pcmcl-ast", "icbhi-pcmcl-beats", "icbhi-beats-ce", "icbhi-beats-pafa", "icbhi-m2dx-s5",
    "icbhi-m2dresp", "icbhi-operact-frozen", "icbhi-lungsrad-dualpaper", "icbhi-cnn6-ce", "icbhi-resnet18-ce", "icbhi-resnet34-ce",
    "icbhi-effb0-ce", "icbhi-cnn6-sgscl", "icbhi-resnet18-sgscl", "icbhi-effb0-sgscl", "icbhi-sclmeta-mscl", "icbhi-protonet",
    "icbhi-dt-paper", "icbhi-amsd", "icbhi-lorarsc", "icbhi-audioclap", "icbhi-bts",
]


def ensure_merged_index() -> pathlib.Path:
    """.datasets/sprsound/index/index_cycle_icbhi4.csv — index_cycle.csv 의 label 을 7→4 병합한 파생 사본 (없으면 생성)."""
    import pandas as pd
    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4
    src = ROOT / ".datasets" / "sprsound" / "index" / "index_cycle.csv"
    dst = src.with_name("index_cycle_icbhi4.csv")
    if not dst.exists():
        df = pd.read_csv(src); df["label"] = [SPRSOUND_7_TO_4[int(v)] for v in df["label"]]
        df.to_csv(dst, index=False)
    return dst


def family_of(run: str) -> str:
    return re.sub(r"-s\d+$", "", run)


def _redirect_cache_dirs(obj, target: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "cache_dir" and isinstance(v, str):
                obj[k] = target
            else:
                _redirect_cache_dirs(v, target)
    elif isinstance(obj, list):
        for v in obj:
            _redirect_cache_dirs(v, target)


def make_ood_data_cfg(data_cfg: dict, family: str) -> tuple[dict, str]:
    """run 의 data config → SPRSound OOD 뷰. 반환 (cfg, chain_note)."""
    cfg = copy.deepcopy(data_cfg)
    p = cfg.get("params", {})
    dm = p.get("datamodule_name", "index")
    if dm == "respirenet":
        # RespireNet: ICBHI txt 주석 전용 데이터모듈 → test 셋만 SPRSound 병합 라벨 인덱스에서 만든다(정규화 통계·청킹·이미지 체인은 학습과 동일)
        p["ood_index_csv"] = str(ROOT / ".datasets" / "sprsound" / "index" / "index_cycle_icbhi4.csv")
        p["ood_data_dir"] = str(ROOT / ".datasets" / "sprsound" / "Classification")
        p["ood_split"] = "test"
        return cfg, "chain=icbhi_respirenet (4 kHz · 8 s chunks · magma image · cycle voting) applied to SPRSound"
    if dm == "bts_precomputed":
        # ICBHI 학습은 저자 사전계산 CLAP 피처 → SPRSound 는 BTS-CARD OOD 와 같은 툴킷 체인(sprsound_btscard_train: 48 kHz, meta7)
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "rsc_toolkit/configs/data/sprsound_btscard_train.yaml"), resolve=False)
        cfg["params"]["project_root"] = str(ROOT)
        p = cfg["params"]; note = "chain=sprsound_btscard_train (BTS-CARD OOD toolkit view)"
    else:
        note = f"chain={cfg.get('name', '?')} (ICBHI recipe chain applied to SPRSound)"
    p.update(SPRS)
    p.pop("split_file", None)
    p["train_splits"] = ["test"]; p["val_splits"] = ["test"]; p["test_splits"] = ["test"]
    p["shuffle_train"] = False
    if dm == "lungmix":
        p.setdefault("lungmix", {})["label_map"] = "none"      # CSV 가 이미 4-class
    if dm == "qlung":
        p["label_map"] = None
    if dm == "amsd":
        p["train_splits"] = ["train"]                           # AMSD 는 split 이름으로 train/val 을 구분한다
    target = str(CACHE_ROOT / str(cfg.get("name", dm)))
    if "features" in p and isinstance(p["features"], dict):
        p["features"]["cache_dir"] = target
    _redirect_cache_dirs(p, target)
    return cfg, note


def _iter_index_datasets(obj, seen=None, depth=0):
    from rsc_toolkit.datasets.index_dataset import IndexDataset
    seen = seen if seen is not None else set()
    if obj is None or id(obj) in seen or depth > 5:
        return
    seen.add(id(obj))
    if isinstance(obj, IndexDataset):
        yield obj; return
    for attr in ("index_ds", "dataset", "ds", "base", "inner"):
        if hasattr(obj, attr):
            yield from _iter_index_datasets(getattr(obj, attr), seen, depth + 1)
    if hasattr(obj, "datasets"):
        for d in obj.datasets:
            yield from _iter_index_datasets(d, seen, depth + 1)
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            if isinstance(v, torch.utils.data.Dataset):
                yield from _iter_index_datasets(v, seen, depth + 1)


def remap_labels(datamodule) -> list[int]:
    """setup 후 val/test/train 의 IndexDataset df.label 을 7→4 로 (이미 0..3 이면 그대로). val 의 4-class 개수를 반환."""
    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4
    counts = None
    for role in ("val_ds", "test_ds", "train_ds"):
        for ds in _iter_index_datasets(getattr(datamodule, role, None)):
            if not getattr(ds, "_ood_remapped", False):
                labels = ds.df["label"]
                if labels.max() > 3:
                    ds.df["label"] = [SPRSOUND_7_TO_4[int(v)] for v in labels]
                ds._ood_remapped = True
            if role == "val_ds" and counts is None:
                counts = [int((ds.df["label"] == c).sum()) for c in range(4)]
    return counts or []


def evaluate(cfg: dict, ckpt: pathlib.Path, family: str, sandbox: pathlib.Path, run: str) -> dict:
    import lightning as L
    from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
    from rsc_toolkit.trainer.factories.build_loss import build_loss
    from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
    from rsc_toolkit.trainer.factories.build_model import build_model
    from rsc_toolkit.trainer.factories.build_trainer import build_trainer

    info: dict = {}
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg["seed"]), workers=True)
    data_cfg, note = make_ood_data_cfg(cfg["data"], family)
    info["chain"] = note
    model = build_model(copy.deepcopy(cfg["model"]))
    loss_func = build_loss(copy.deepcopy(cfg["loss"]))
    train_fns, val_fns = build_train_val_metric_factories(copy.deepcopy(cfg.get("metrics", {})))
    datamodule = build_datamodule(copy.deepcopy(data_cfg))
    trainer = build_trainer(copy.deepcopy(cfg["trainer"]))

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    vw._adapt_buffer_shapes(model, sd)
    model.load_state_dict(sd, strict=True); info["load"] = "strict_ok"
    if hasattr(model, "meta_dialect"):           # BTS: SPRSound 문장 템플릿 (BTS-CARD OOD 프로토콜과 동일)
        model.meta_dialect = "sprsound"; info["meta_dialect"] = "sprsound"

    adapter_cfg = cfg.get("adapter") or {}
    if adapter_cfg.get("name") == "split_iter":
        from rsc_toolkit.trainer.adapters.split_iter_adapter import SplitIterAdapter
        pl_module = SplitIterAdapter(rsc_model=model, loss_func=loss_func, train_metric_fns=train_fns, val_metric_fns=val_fns,
                                     optimizer_cfg=cfg["optimizer"], **(adapter_cfg.get("params") or {}))
        info["note"] = "split_iter: 평가 시 나머지 배치 폐기(원본 quirk)"
    else:
        pl_module = PLAdapter(rsc_model=model, loss_func=loss_func, train_metric_fns=train_fns, val_metric_fns=val_fns,
                              optimizer_cfg=cfg["optimizer"])

    orig_setup = datamodule.setup
    def _setup(stage=None):
        orig_setup(None)
        if getattr(datamodule, "ood", None):                 # RespireNet 인덱스 경로: 사이클 단위 라벨 개수
            info["val_counts"] = list(datamodule.test_ds.cycle_label_counts)
        else:
            info["val_counts"] = remap_labels(datamodule)
    datamodule.setup = _setup  # type: ignore[assignment]
    out = trainer.validate(pl_module, datamodule=datamodule, verbose=False)
    metrics = {}
    for d in out or []:
        metrics.update({k: float(v) for k, v in d.items()})
    metrics.update({k: float(v) for k, v in trainer.callback_metrics.items()})
    if "vote/icbhi_score" in metrics:                        # RespireNet: 사이클 다수결 지표(청크 지표가 아니라)
        metrics = {**metrics, **{k: metrics[v] for k, v in vw.VOTE_MAP.items() if v in metrics}}
    info["metrics"] = {k: metrics[k] for k in ("S_p", "S_e", "icbhi_score") if k in metrics}
    info["all_metrics"] = metrics
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default=None, help="쉼표 구분 family 목록 (기본: DEFAULT_FAMILIES)")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--runs", default=None, help="쉼표 구분 run 이름 목록 (family/seed 대신 명시)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-write-run", action="store_true", help="outputs/<group>/<run>/ood.json 을 쓰지 않는다 (검증용)")
    ap.add_argument("--max-temp", type=int, default=80)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ensure_merged_index()
    sandbox = out / "sandbox"; vw.make_sandbox(sandbox)
    fams = [f for f in (args.families.split(",") if args.families else DEFAULT_FAMILIES) if f]
    seeds = [int(s) for s in args.seeds.split(",")]
    import csv
    master = list(csv.DictReader(open(ROOT / "docs" / "results_master.csv")))
    by_run = {r["run_name"]: r for r in master}
    log = open(out / "results.jsonl", "a")
    jobs = [(family_of(r), r) for r in args.runs.split(",") if r] if args.runs else [(f, f"{f}-s{s}") for f in fams for s in seeds]
    for fam, run in jobs:
        if True:
            row = by_run.get(run)
            rec = {"family": fam, "run": run, "status": "ERROR", "ts": time.time()}
            t0 = time.time()
            try:
                if row is None:
                    raise KeyError(f"{run} 이 results_master 에 없다")
                run_dir = ROOT / row["run_dir"]
                if (run_dir / "ood.json").exists() and not args.no_write_run:
                    rec.update(status="SKIP", note="ood.json 이미 있음"); continue
                cfg, source = vw.resolve_config(row, master)
                if cfg is None:
                    rec.update(status="SKIP", note="설정 스냅샷 없음"); continue
                cfg, dropped = vw.sanitize_config(cfg, sandbox, run)
                rec["config_source"], rec["dropped_callbacks"] = source, dropped
                vw.wait_for_cool_gpu(args.max_temp, 70)
                info = evaluate(cfg, run_dir / "best.pth", fam, sandbox, run)
                rec.update({k: v for k, v in info.items() if k != "all_metrics"})
                m = info.get("metrics", {})
                if not m or len(m) < 3:
                    rec["status"] = "ERROR"; rec["error"] = f"metrics missing: {list(info.get('all_metrics', {}))[:10]}"
                elif info.get("val_counts") != EXPECTED_COUNTS:
                    rec["status"] = "COUNT_MISMATCH"
                else:
                    rec["status"] = "OK"
                    if not args.no_write_run:
                        payload = {"inter": {**m, "n": 1429, "class_counts": info["val_counts"]},
                                   "protocol": "ICBHI-trained best.pth → SPRSound 2022 inter-patient test, 7→4 merged; "
                                               "recipe's ICBHI preprocessing chain applied to SPRSound",
                                   "chain": info.get("chain"), "generated_by": "scripts/eval_ood_sprsound.py", "generated": time.strftime("%Y-%m-%d %H:%M")}
                        if info.get("note"):
                            payload["note"] = info["note"]
                        (run_dir / "ood.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:800]; rec["traceback"] = traceback.format_exc()[-2000:]
            finally:
                rec["seconds"] = round(time.time() - t0, 1)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
                m = rec.get("metrics") or {}
                print(f"{run:32s} {rec['status']:14s} " + (f"Sp {m['S_p']:.2f} Se {m['S_e']:.2f} Sc {m['icbhi_score']:.2f}" if m else rec.get("error", rec.get("note", ""))[:120]) + f"  ({rec['seconds']}s)", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
