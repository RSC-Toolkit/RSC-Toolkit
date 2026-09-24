"""
Zero-shot evaluation of ICBHI-trained checkpoints on the SPRSound 2022 inter-patient test set.

Each run is rebuilt from its config snapshot with the same factories used in training (best.pth loaded strictly); only the
dataset is switched to SPRSound. The recipe's ICBHI preprocessing chain is applied to SPRSound unchanged, and the SPRSound
labels are mapped to the ICBHI 4-class space (Normal -> 0, Fine/Coarse Crackle -> 1, Wheeze -> 2, Both -> 3; Rhonchi and
Stridor events are excluded, and the 2022 inter-patient test set contains neither). Metrics are the ICBHI 4-class
S_p / S_e / Score on the 1,429 test events.

Prerequisite: the SPRSound index (`rsctk datasets download sprsound` and `rsctk datasets build-index sprsound`).

  python scripts/eval_zero_shot_sprsound.py outputs/patchmix/icbhi-patchmix-ce
  python scripts/eval_zero_shot_sprsound.py paper_weights/icbhi-cnn6-ce-s1 --sprsound-dir .datasets/sprsound --out zs.jsonl

Nothing is written into the run directories. If a run directory contains ood.json (the toolkit's recorded zero-shot result),
the new result is compared with it (PASS when |Δ| <= --tol).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pathlib
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsc_toolkit.evaluate import (  # noqa: E402
    STD_KEYS, VOTE_MAP, _chdir, apply_overrides, load_checkpoint, make_sandbox, resolve_run_config, sanitize_config,
)

EXPECTED_COUNTS = [1040, 83, 305, 1]   # normal / crackle / wheeze / both in the 2022 inter-patient test set
logger = logging.getLogger("eval_zero_shot_sprsound")


def ensure_4class_index(sprsound_dir: pathlib.Path) -> pathlib.Path:
    """Derived index `index/index_cycle_icbhi4.csv`: index_cycle.csv with Rhonchi/Stridor rows removed and labels mapped 7 -> 4."""
    import pandas as pd

    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4, apply_label_map

    src = sprsound_dir / "index" / "index_cycle.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run `rsctk datasets build-index sprsound --data_dir {sprsound_dir}` first")
    dst = src.with_name("index_cycle_icbhi4.csv")
    derived, dropped = apply_label_map(pd.read_csv(src), SPRSOUND_7_TO_4, where="index_cycle_icbhi4")
    derived = derived.drop(columns=["label_original"])
    if dst.exists():
        old = pd.read_csv(dst)
        if len(old) == len(derived) and int(pd.to_numeric(old["label"], errors="coerce").max()) <= 3:
            return dst
    derived.to_csv(dst, index=False)
    logger.info("wrote %s (%d rows, %d Rhonchi/Stridor rows excluded)", dst, len(derived), dropped)
    return dst


def _redirect_cache_dirs(obj: Any, target: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "cache_dir" and isinstance(v, str):
                obj[k] = target
            else:
                _redirect_cache_dirs(v, target)
    elif isinstance(obj, list):
        for v in obj:
            _redirect_cache_dirs(v, target)


def make_zero_shot_data_cfg(data_cfg: dict, sprsound_dir: pathlib.Path, cache_root: pathlib.Path) -> tuple[dict, str]:
    """Turn a run's ICBHI data config into the SPRSound zero-shot view. Returns (config, description of the chain)."""
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(data_cfg)
    p = cfg.get("params", {})
    dm = p.get("datamodule_name", "index")
    index_4class = sprsound_dir / "index" / "index_cycle_icbhi4.csv"
    if dm == "respirenet":
        # RespireNet reads ICBHI txt annotations; only its test set is built from the SPRSound 4-class index
        # (normalisation statistics, chunking and image chain stay as in training).
        p["ood_index_csv"] = str(index_4class)
        p["ood_data_dir"] = str(sprsound_dir / "Classification")
        p["ood_split"] = "test"
        return cfg, "RespireNet chain (4 kHz, 8 s chunks, magma image, cycle voting) applied to SPRSound"
    if dm == "bts_precomputed":
        # ICBHI training used the authors' precomputed CLAP features; SPRSound uses the toolkit's BTS-CARD chain.
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "rsc_toolkit/configs/data/sprsound_btscard_train.yaml"), resolve=False)
        cfg["params"]["project_root"] = str(ROOT)
        p = cfg["params"]
        note = "sprsound_btscard_train chain"
    else:
        note = f"{cfg.get('name', dm)} chain (ICBHI recipe) applied to SPRSound"
    p.update({"dataset_name": "sprsound", "data_dir": str(sprsound_dir), "raw_data_subdir": "Classification",
              "index_filename": "index_cycle_icbhi4.csv"})
    p.pop("split_file", None)
    p["train_splits"], p["val_splits"], p["test_splits"] = ["test"], ["test"], ["test"]
    p["shuffle_train"] = False
    if dm == "lungmix":
        p.setdefault("lungmix", {})["label_map"] = "none"      # the derived index is already 4-class
    if dm == "qlung":
        p["label_map"] = None
    if dm == "amsd":
        p["train_splits"] = ["train"]                          # AMS-D tells train and validation apart by split name
    target = str(cache_root / str(cfg.get("name", dm)))
    if isinstance(p.get("features"), dict):
        p["features"]["cache_dir"] = target
    _redirect_cache_dirs(p, target)
    return cfg, note


def _iter_index_datasets(obj, seen=None, depth=0):
    import torch

    from rsc_toolkit.datasets.index_dataset import IndexDataset

    seen = seen if seen is not None else set()
    if obj is None or id(obj) in seen or depth > 5:
        return
    seen.add(id(obj))
    if isinstance(obj, IndexDataset):
        yield obj
        return
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


def remap_labels(datamodule) -> List[int]:
    """After setup, map any remaining 7-class labels to 4 classes; return the 4-class counts of the evaluation set."""
    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4, apply_label_map

    counts = None
    for role in ("val_ds", "test_ds", "train_ds"):
        for ds in _iter_index_datasets(getattr(datamodule, role, None)):
            if not getattr(ds, "_zero_shot_remapped", False):
                if int(ds.df["label"].max()) > 3:
                    ds.df, _ = apply_label_map(ds.df, SPRSOUND_7_TO_4, where=f"zero-shot {role}")
                ds._zero_shot_remapped = True
            if role == "val_ds" and counts is None:
                counts = [int((ds.df["label"] == c).sum()) for c in range(4)]
    return counts or []


def build_and_validate_zero_shot(cfg: dict, ckpt: pathlib.Path, sprsound_dir: pathlib.Path, cache_root: pathlib.Path) -> Dict[str, Any]:
    import lightning as L

    from rsc_toolkit.trainer.factories.build_adapter import build_adapter
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
    from rsc_toolkit.trainer.factories.build_loss import build_loss
    from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
    from rsc_toolkit.trainer.factories.build_model import build_model
    from rsc_toolkit.trainer.factories.build_trainer import build_trainer

    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg["seed"]), workers=True)
    data_cfg, chain = make_zero_shot_data_cfg(cfg["data"], sprsound_dir, cache_root)
    model = build_model(copy.deepcopy(cfg["model"]))
    loss_func = build_loss(copy.deepcopy(cfg["loss"]))
    train_fns, val_fns = build_train_val_metric_factories(copy.deepcopy(cfg.get("metrics", {})))
    datamodule = build_datamodule(copy.deepcopy(data_cfg))
    trainer = build_trainer(copy.deepcopy(cfg["trainer"]))

    info: Dict[str, Any] = {"chain": chain, **load_checkpoint(model, ckpt)}
    if info["load"] != "strict_ok":
        return info
    if hasattr(model, "meta_dialect"):          # BTS: SPRSound metadata sentence template
        model.meta_dialect = "sprsound"
    pl_module = build_adapter(cfg.get("adapter"), rsc_model=model, loss_func=loss_func, optimizer_cfg=cfg["optimizer"],
                              train_metric_fns=train_fns, val_metric_fns=val_fns)

    orig_setup = datamodule.setup

    def _setup(stage=None):
        orig_setup(None)
        if getattr(datamodule, "ood", None):    # RespireNet: cycle-level label counts
            info["class_counts"] = list(datamodule.test_ds.cycle_label_counts)
        else:
            info["class_counts"] = remap_labels(datamodule)

    datamodule.setup = _setup  # type: ignore[assignment]
    out = trainer.validate(pl_module, datamodule=datamodule, verbose=False)
    metrics: Dict[str, float] = {}
    for d in out or []:
        metrics.update({k: float(v) for k, v in d.items()})
    metrics.update({k: float(v) for k, v in trainer.callback_metrics.items()})
    if "vote/icbhi_score" in metrics:            # RespireNet reports cycle-voting metrics
        metrics.update({k: metrics[v] for k, v in VOTE_MAP.items() if v in metrics})
    info["metrics"] = {k: metrics[k] for k in STD_KEYS if k in metrics}
    return info


def evaluate_zero_shot(run_dir: pathlib.Path, *, project_root: pathlib.Path, sprsound_dir: pathlib.Path, sandbox: pathlib.Path,
                       hydra_dir: Optional[pathlib.Path], device: str, tol: float, overrides: List[str] = ()) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"run": run_dir.name, "run_dir": str(run_dir), "status": "ERROR"}
    ckpt = run_dir / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    cfg, source = resolve_run_config(run_dir, project_root, hydra_dir=hydra_dir)
    rec["config_source"] = source
    if cfg is None:
        rec.update(status="SKIP", note="no config snapshot and no run log — pass --hydra-dir")
        return rec
    cfg, rec["dropped_callbacks"] = sanitize_config(cfg, sandbox, run_dir.name, project_root)
    cfg = apply_overrides(cfg, list(overrides))
    if device.startswith("cpu"):
        cfg.setdefault("trainer", {}).setdefault("params", {}).update(accelerator="cpu", devices=1)
    cache_root = project_root / "downloaded_datasets" / "features_cache_ood"
    with _chdir(sandbox):
        info = build_and_validate_zero_shot(cfg, ckpt, sprsound_dir, cache_root)
    rec.update(info)
    m = info.get("metrics") or {}
    if info.get("load") != "strict_ok":
        rec["status"] = "LOAD_FAIL"
    elif len(m) < 3:
        rec["status"] = "ERROR"
    elif info.get("class_counts") != EXPECTED_COUNTS:
        rec["status"] = "COUNT_MISMATCH"
    else:
        rec["status"] = "OK"
        recorded = run_dir / "ood.json"
        if recorded.exists():
            recorded_json = json.loads(recorded.read_text())
            # {"inter": {...}} from this script; BTS-CARD records {"sprsound_toolkit": {"alpha=0.0": {...}}}
            ref = recorded_json.get("inter") or (recorded_json.get("sprsound_toolkit") or {}).get("alpha=0.0") or {}
            diffs = {k: round(abs(m[k] - float(ref[k])), 4) for k in STD_KEYS if k in ref}
            rec["diffs_vs_ood_json"] = diffs
            rec["status"] = "PASS" if diffs and max(diffs.values()) <= tol else "MISMATCH"
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run directories trained on ICBHI (containing best.pth), e.g. outputs/<group>/<run>")
    ap.add_argument("--project-root", default=None, help="base for relative paths (default: current directory)")
    ap.add_argument("--sprsound-dir", default=".datasets/sprsound", help="SPRSound data directory (with index/index_cycle.csv)")
    ap.add_argument("--hydra-dir", default=None, help="config snapshot directory (only with a single run directory)")
    ap.add_argument("--sandbox", default=None, help="directory for all writes (default: <project-root>/.rsc_eval_sandbox; must be outside outputs/)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--override", action="append", default=[], help="OmegaConf dotlist applied to the run config before the SPRSound switch, e.g. data.params.batch_size=16 (repeatable)")
    ap.add_argument("--tol", type=float, default=0.05, help="allowed |Δ| against an existing ood.json")
    ap.add_argument("--out", default=None, help="append one JSON line per run to this file")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    project_root = pathlib.Path(args.project_root or os.getcwd()).resolve()
    sprsound_dir = pathlib.Path(args.sprsound_dir)
    sprsound_dir = (sprsound_dir if sprsound_dir.is_absolute() else project_root / sprsound_dir).resolve()
    sandbox = pathlib.Path(args.sandbox).resolve() if args.sandbox else project_root / ".rsc_eval_sandbox"
    if (project_root / "outputs") in sandbox.parents:
        raise SystemExit(f"sandbox {sandbox} is inside outputs/ — choose a directory outside outputs/")
    if args.hydra_dir and len(args.run_dirs) > 1:
        raise SystemExit("--hydra-dir can only be used with a single run directory")
    ensure_4class_index(sprsound_dir)
    make_sandbox(sandbox, project_root)

    failed = 0
    for rd in args.run_dirs:
        run_dir = pathlib.Path(rd)
        run_dir = (run_dir if run_dir.is_absolute() else project_root / run_dir).resolve()
        t0 = time.time()
        try:
            rec = evaluate_zero_shot(run_dir, project_root=project_root, sprsound_dir=sprsound_dir, sandbox=sandbox,
                                     hydra_dir=pathlib.Path(args.hydra_dir) if args.hydra_dir else None, device=args.device, tol=args.tol,
                                     overrides=args.override)
        except Exception as exc:  # keep going with the next run
            rec = {"run": run_dir.name, "run_dir": str(run_dir), "status": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:800], "traceback": traceback.format_exc()[-2000:]}
        rec["seconds"] = round(time.time() - t0, 1)
        m = rec.get("metrics") or {}
        summary = " ".join(f"{k}={m[k]:.2f}" for k in STD_KEYS if k in m) or rec.get("error", rec.get("note", ""))[:200]
        print(f"{rec['run']}: {rec['status']} {summary} ({rec['seconds']} s)", flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        failed += rec["status"] not in ("OK", "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
