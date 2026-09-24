"""Checkpoint re-evaluation — `rsc-toolkit-eval`.

Rebuilds a training run directory (`outputs/<group>/<run>/best.pth`) from that run's config snapshot
(`outputs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml`, or else the resolved config printed in the run log) with **the same
factories as in training**, loads the state_dict strictly, evaluates the validation split with `trainer.validate`, and returns the metrics.

Principles
- Training outputs are **read only**: the process CWD moves to a sandbox (only input directories are symlinked), callbacks
  that write checkpoints/json are dropped or redirected to the sandbox, and loggers are disabled. The feature cache's chain ledger (`_chain.json`) is only read too.
- The snapshot config is used as is, but can be overridden with `--override k=v` (OmegaConf dotlist) — e.g. a different batch size or split.

    rsc-toolkit-eval outputs/patchmix/icbhi-patchmix-ce-s1                 # compared against best.json automatically
    rsc-toolkit-eval outputs/qlung/icbhi-qlung-s1 --override data.params.batch_size=16 --out eval.json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import json
import logging
import os
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Top-level directories used only as inputs — exposed in the sandbox as symlinks. outputs/ is deliberately excluded.
READ_ONLY_ROOTS: Tuple[str, ...] = ("rsc_toolkit", ".datasets", "downloaded_datasets", "pretrained_models", "sample_data")
# Prefixes of relative paths in the config to make absolute (outputs/ appears only in **inputs** such as encoder_ckpt;
# callback dirpaths are handled separately by sanitize_config).
ABS_PREFIXES: Tuple[str, ...] = tuple(f"{r}/" for r in READ_ONLY_ROOTS) + ("outputs/",)
# Callbacks kept for validate — only those needed to reproduce metrics/loss, or harmless ones. All others (checkpoint/json
# saving, MA/EMA, LR schedules, sampler epochs, etc.) are dropped.
KEEP_CALLBACKS = frozenset({"AMSDEpochCallback", "DomainTransferStages", "ProtoNetClassWeight", "RespireNetVoteEval"})
STD_KEYS: Tuple[str, ...] = ("S_p", "S_e", "icbhi_score")
VOTE_MAP = {"S_p": "vote/S_p", "S_e": "vote/S_e", "icbhi_score": "vote/icbhi_score", "cycle_acc": "vote/cycle_acc"}


# ----------------------------------------------------------------------------- config recovery
def load_snapshot_config(hydra_dir: pathlib.Path, project_root: pathlib.Path) -> dict:
    """Read `<hydra_dir>/.hydra/config.yaml`. `${hydra:runtime.cwd}` cannot be resolved outside the hydra runtime, so it is replaced with project_root."""
    import yaml
    from omegaconf import OmegaConf

    text = (pathlib.Path(hydra_dir) / ".hydra" / "config.yaml").read_text()
    text = text.replace("${hydra:runtime.cwd}", str(project_root))
    cfg = OmegaConf.create(yaml.safe_load(text))
    return OmegaConf.to_container(cfg, resolve=True)


def find_run_log(run_name: str, project_root: pathlib.Path) -> Optional[pathlib.Path]:
    hits = sorted((pathlib.Path(project_root) / "outputs").glob(f"*/logs/{run_name}.log"))
    return hits[-1] if hits else None


def load_log_config(log_path: pathlib.Path) -> Optional[dict]:
    """Parse the `OmegaConf.to_yaml(cfg, resolve=True)` block that train.py prints at startup (the last block)."""
    import yaml
    from omegaconf import OmegaConf

    lines = pathlib.Path(log_path).read_text(errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln == "model:"]
    if not starts:
        return None
    block = []
    for ln in lines[starts[-1]:]:
        # Only top-level keys (lowercase key:) or indented lines belong to the YAML block. Stop at log lines, warnings, or Lightning output.
        if ln.startswith((" ", "-")) or re.match(r"^[a-z_]+:( .*)?$", ln):
            block.append(ln)
        else:
            break
    cfg = yaml.safe_load("\n".join(block))
    if not isinstance(cfg, dict) or "model" not in cfg or "data" not in cfg:
        return None
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def _snapshot_run_name(config_yaml: pathlib.Path) -> Optional[str]:
    """run.name of the snapshot (unresolved). If it is the default `${data.name}-${model.name}`, assemble it from the two names."""
    from omegaconf import OmegaConf

    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_yaml), resolve=False)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    name = ((raw.get("run") or {}).get("name")) if isinstance(raw.get("run"), dict) else None
    if isinstance(name, str) and "${" in name:
        data_name = (raw.get("data") or {}).get("name") if isinstance(raw.get("data"), dict) else None
        model_name = (raw.get("model") or {}).get("name") if isinstance(raw.get("model"), dict) else None
        name = name.replace("${data.name}", str(data_name)).replace("${model.name}", str(model_name))
    return name if isinstance(name, str) else None


def find_snapshot_dirs(run_name: str, project_root: pathlib.Path) -> List[pathlib.Path]:
    """Snapshot directories under `outputs/*/*/.hydra/config.yaml` with the same run.name (sorted by path)."""
    outputs = pathlib.Path(project_root) / "outputs"
    found = []
    for cfg_yaml in sorted(outputs.glob("*/*/.hydra/config.yaml")):
        if _snapshot_run_name(cfg_yaml) == run_name:
            found.append(cfg_yaml.parent.parent)
    return found


def pick_snapshot(snapshot_dirs: Sequence[pathlib.Path], best_pth: pathlib.Path) -> Optional[pathlib.Path]:
    """With several snapshots of the same run.name, pick the latest one **started before best.pth was saved** (directory name = start time); otherwise the last one."""
    if not snapshot_dirs:
        return None
    try:
        saved = datetime.datetime.fromtimestamp(pathlib.Path(best_pth).stat().st_mtime)
    except OSError:
        return snapshot_dirs[-1]
    before = []
    for d in snapshot_dirs:
        try:
            started = datetime.datetime.strptime(f"{d.parent.name} {d.name}", "%Y-%m-%d %H-%M-%S")
        except ValueError:
            continue
        if started <= saved:
            before.append(d)
    return before[-1] if before else snapshot_dirs[-1]


def resolve_run_config(run_dir: pathlib.Path, project_root: pathlib.Path, *, hydra_dir: Optional[pathlib.Path] = None,
                       ckpt_name: str = "best.pth") -> Tuple[Optional[dict], str]:
    """Recover the config of a run directory: given hydra_dir → run_dir/.hydra → snapshot with the same run.name → run log. Returns (cfg, source)."""
    run_dir = pathlib.Path(run_dir)
    if hydra_dir is not None:
        return load_snapshot_config(pathlib.Path(hydra_dir), project_root), f"hydra:{hydra_dir}"
    if (run_dir / ".hydra" / "config.yaml").exists():
        return load_snapshot_config(run_dir, project_root), f"hydra:{run_dir}"
    snaps = find_snapshot_dirs(run_dir.name, project_root)
    chosen = pick_snapshot(snaps, run_dir / ckpt_name)
    if chosen is not None:
        return load_snapshot_config(chosen, project_root), f"hydra:{chosen}"
    log_path = find_run_log(run_dir.name, project_root)
    if log_path is not None:
        cfg = load_log_config(log_path)
        if cfg is not None:
            return cfg, f"log:{log_path}"
    return None, "none"


def absolutize(obj: Any, project_root: pathlib.Path, skip_keys: Iterable[str] = ()) -> Any:
    """Turn relative path strings (READ_ONLY_ROOTS / outputs prefixes) into absolute paths under project_root. skip_keys subtrees are skipped."""
    skip = set(skip_keys)
    if isinstance(obj, dict):
        return {k: (v if k in skip else absolutize(v, project_root, skip)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [absolutize(v, project_root, skip) for v in obj]
    if isinstance(obj, str) and obj.startswith(ABS_PREFIXES):
        return str(pathlib.Path(project_root) / obj)
    return obj


def sanitize_config(cfg: dict, sandbox: pathlib.Path, run_name: str, project_root: pathlib.Path,
                    keep_callbacks: Iterable[str] = KEEP_CALLBACKS) -> Tuple[dict, List[str]]:
    """Convert to an evaluation-only config: drop resume, make paths absolute, drop writing callbacks (kept ones get a sandbox dirpath), disable loggers and checkpointing."""
    keep = set(keep_callbacks)
    cfg = copy.deepcopy(cfg)
    cfg.pop("resume", None)
    tp = cfg.setdefault("trainer", {}).setdefault("params", {})
    callbacks = tp.get("callbacks") or []
    cfg["trainer"]["params"] = {k: v for k, v in tp.items() if k != "callbacks"}
    cfg = absolutize(cfg, project_root)
    tp = cfg["trainer"]["params"]

    kept, dropped = [], []
    for cb in callbacks:
        name = str(cb.get("_target_", "")).rsplit(".", 1)[-1]
        if name in keep:
            cb = dict(cb)
            if "dirpath" in cb:  # RespireNetVoteEval — writes best.pth/best.json, so redirect to the sandbox
                cb["dirpath"] = str(pathlib.Path(sandbox) / "callback_out" / run_name)
            kept.append(cb)
        else:
            dropped.append(name)
    tp["callbacks"] = kept
    tp["logger"] = False
    tp["enable_checkpointing"] = False
    tp["enable_progress_bar"] = False
    tp["enable_model_summary"] = False
    tp["default_root_dir"] = str(pathlib.Path(sandbox) / "lightning")
    return cfg, dropped


def apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict:
    """Merge a `k.sub=v` dotlist with OmegaConf (values parsed as YAML: 8 → int, true → bool)."""
    if not overrides:
        return cfg
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(merged, resolve=True)


# ----------------------------------------------------------------------------- sandbox and cache ledger
def make_sandbox(path: pathlib.Path, project_root: pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in READ_ONLY_ROOTS:
        src, dst = pathlib.Path(project_root) / name, path / name
        if not src.exists():
            continue
        if dst.is_symlink() and dst.resolve() != src.resolve():
            dst.unlink()                      # relink so that a sandbox left by another checkout does not silently read from there
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    (path / "lightning").mkdir(exist_ok=True)
    (path / "callback_out").mkdir(exist_ok=True)


def install_readonly_chain_guard(notes: list):
    """Make FeatureCache._check_audio_chain read-only. Returns a function that restores it.

    The original **writes a new ledger (_chain.json) into the real cache directory** if none exists, and raises on a mismatch.
    Evaluation must only read the cache, so writing is removed and mismatches are recorded in `notes` instead of raising
    (the waveform kind uses no cache at all, so a mismatch does not affect evaluation — the record shows that as well)."""
    from rsc_toolkit.datasets import feature_cache as fc_mod

    original = fc_mod.FeatureCache._check_audio_chain

    def _readonly_check(self, chain):
        dataset = str(chain.pop("dataset_name", "?"))
        path = self.features_dir / "_chain.json"
        if not path.exists():
            return
        try:
            known = (json.loads(path.read_text()) or {}).get(dataset)
        except Exception:
            return
        if known is not None and known != chain:
            n_files = sum(1 for p in self.features_dir.iterdir() if p.name != "_chain.json")
            notes.append({"cache": str(self.features_dir), "dataset": dataset, "ledger": known,
                          "requested": chain, "cached_files": n_files})

    fc_mod.FeatureCache._check_audio_chain = _readonly_check  # type: ignore[method-assign]

    def restore() -> None:
        fc_mod.FeatureCache._check_audio_chain = original  # type: ignore[method-assign]

    return restore


@contextlib.contextmanager
def readonly_chain_guard(notes: Optional[list] = None) -> Iterator[list]:
    """With-block form of `install_readonly_chain_guard` — restores the original behavior on exit."""
    notes = [] if notes is None else notes
    restore = install_readonly_chain_guard(notes)
    try:
        yield notes
    finally:
        restore()


@contextlib.contextmanager
def _chdir(path: pathlib.Path) -> Iterator[None]:
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ----------------------------------------------------------------------------- load and evaluate
# Buffers whose size depends on the number of training samples (AMS-D per-sample state) — only these names are reshaped to the ckpt.
# Any other shape mismatch (BN running stats, pos_embed, etc.) means a different model configuration, so the strict load must fail.
ADAPTABLE_BUFFER_SUFFIXES = ("all_predictions", "ids_masked")


def adapt_buffer_shapes(model, sd: dict) -> List[str]:
    """If a ckpt buffer shape differs from the new model (AMSD per-sample buffers), reshape only that buffer to the ckpt shape."""
    import torch

    changed = []
    buffers = dict(model.named_buffers())
    for name, tensor in sd.items():
        cur = buffers.get(name)
        if cur is None or tuple(cur.shape) == tuple(tensor.shape):
            continue
        if name.rpartition(".")[2] not in ADAPTABLE_BUFFER_SUFFIXES:
            continue
        mod_name, _, attr = name.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model
        mod._buffers[attr] = torch.zeros_like(tensor, device=cur.device)
        changed.append(f"{name}: {tuple(cur.shape)} -> {tuple(tensor.shape)}")
    return changed


def load_checkpoint(model, ckpt_path: pathlib.Path) -> dict:
    """Strictly load best.pth (`{"model": state_dict}` or a bare state_dict). Returns an info dict (load: strict_ok|strict_fail)."""
    import torch

    info: Dict[str, Any] = {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    info["buffer_adapted"] = adapt_buffer_shapes(model, sd)
    try:
        model.load_state_dict(sd, strict=True)
        info["load"] = "strict_ok"
    except RuntimeError as exc:
        info["load"] = "strict_fail"
        info["load_error"] = str(exc)[:500]
        try:
            res = model.load_state_dict(sd, strict=False)
            info["load_missing"] = list(res.missing_keys)[:20]
            info["load_unexpected"] = list(res.unexpected_keys)[:20]
        except RuntimeError as exc2:
            # Shape mismatches (buffers/weights) cannot be loaded even with strict=False — record a failure without partial-load info
            info["load_error"] = str(exc2)[:500]
    return info


def build_and_validate(cfg: dict, ckpt_path: pathlib.Path, sandbox: pathlib.Path, run_name: str) -> dict:
    """Build model, loss, metrics, datamodule, and trainer from the (sanitized) config, load the ckpt, and validate. Returns info (load, metrics…)."""
    import lightning as L

    from rsc_toolkit.trainer.factories.build_adapter import build_adapter
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule
    from rsc_toolkit.trainer.factories.build_loss import build_loss
    from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories
    from rsc_toolkit.trainer.factories.build_model import build_model
    from rsc_toolkit.trainer.factories.build_trainer import build_trainer

    seed = cfg.get("seed")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)

    model = build_model(copy.deepcopy(cfg["model"]))
    loss_func = build_loss(copy.deepcopy(cfg["loss"]))
    train_fns, val_fns = build_train_val_metric_factories(copy.deepcopy(cfg.get("metrics", {})))
    datamodule = build_datamodule(copy.deepcopy(cfg["data"]))
    trainer = build_trainer(copy.deepcopy(cfg["trainer"]))

    info = load_checkpoint(model, pathlib.Path(ckpt_path))
    if info["load"] != "strict_ok":
        return info

    pl_module = build_adapter(cfg.get("adapter"), rsc_model=model, loss_func=loss_func, optimizer_cfg=cfg["optimizer"],
                              train_metric_fns=train_fns, val_metric_fns=val_fns)

    # Lightning calls setup("validate") for validate, but toolkit datamodules only know fit/test → map to None (all).
    orig_setup = datamodule.setup
    datamodule.setup = lambda stage=None: orig_setup(None if stage == "validate" else stage)  # type: ignore[assignment]

    out = trainer.validate(pl_module, datamodule=datamodule, verbose=False)
    metrics: Dict[str, float] = {}
    for d in out or []:
        metrics.update({k: float(v) for k, v in d.items()})
    metrics.update({k: float(v) for k, v in trainer.callback_metrics.items()})
    info["metrics"] = metrics
    return info


def compare(expected: dict, metrics: dict, tol: float) -> Tuple[str, dict, Optional[float]]:
    """Compare best.json (expected) with re-evaluated metrics — PASS (|Δ| ≤ tol) · MISMATCH · ERROR (no keys to compare)."""
    if "cycle_acc" in expected:  # RespireNet — cycle-vote metrics
        pairs = {k: (expected.get(k), metrics.get(v)) for k, v in VOTE_MAP.items()}
    else:
        keys = list(STD_KEYS) + [k for k in ("paper_S_p", "paper_S_e", "paper_score", "paper_acc") if k in expected]
        pairs = {k: (expected.get(k), metrics.get(k)) for k in keys}
    diffs: Dict[str, Any] = {}
    missing, worst = [], None
    for k, (e, m) in pairs.items():
        if e is None:
            continue
        if m is None:
            missing.append(k)
            continue
        d = abs(float(e) - float(m))
        diffs[k] = {"expected": float(e), "evaluated": float(m), "diff": d}
        worst = d if worst is None else max(worst, d)
    if not diffs:
        return "ERROR", {"missing_eval_keys": missing}, None
    status = "PASS" if worst <= tol else "MISMATCH"
    if missing:
        diffs["missing_eval_keys"] = missing
    return status, diffs, worst


def evaluate_run(run_dir: pathlib.Path, *, project_root: Optional[pathlib.Path] = None,
                 hydra_dir: Optional[pathlib.Path] = None, ckpt_name: str = "best.pth",
                 overrides: Sequence[str] = (), sandbox: Optional[pathlib.Path] = None,
                 tol: float = 0.05) -> dict:
    """Re-evaluate one run directory. Returned dict: run, config_source, dropped_callbacks, load, metrics, and (if best.json exists) status and diffs."""
    project_root = pathlib.Path(project_root or os.getcwd()).resolve()
    run_dir = pathlib.Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = (project_root / run_dir).resolve()
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    rec: Dict[str, Any] = {"run": run_dir.name, "run_dir": str(run_dir), "ckpt": str(ckpt_path)}

    cfg, source = resolve_run_config(run_dir, project_root, hydra_dir=hydra_dir, ckpt_name=ckpt_name)
    rec["config_source"] = source
    if cfg is None:
        rec["status"] = "SKIP"
        rec["note"] = "no config snapshot and no run log — specify --hydra-dir"
        return rec

    sandbox = pathlib.Path(sandbox) if sandbox else pathlib.Path(project_root) / ".rsc_eval_sandbox"
    if (project_root / "outputs") in sandbox.resolve().parents:
        raise ValueError(f"sandbox {sandbox} is inside outputs/ — callback outputs would mix into the results directory")
    make_sandbox(sandbox, project_root)
    cfg, dropped = sanitize_config(cfg, sandbox, run_dir.name, project_root)
    cfg = apply_overrides(cfg, overrides)
    rec["dropped_callbacks"] = dropped
    rec["overrides"] = list(overrides)

    with readonly_chain_guard() as notes, _chdir(sandbox):
        info = build_and_validate(cfg, ckpt_path, sandbox, run_dir.name)
    if notes:
        rec["cache_chain_mismatch"] = list(notes)
    rec.update(info)
    if info.get("load") != "strict_ok":
        rec["status"] = "LOAD_FAIL"
        return rec

    best_json = run_dir / "best.json"
    if best_json.exists():
        expected = json.loads(best_json.read_text())
        status, diffs, worst = compare(expected, info["metrics"], tol)
        # With overrides the protocol may have changed, so no PASS/MISMATCH verdict is given ("OK"), but the
        # comparison (diffs, max_abs_diff) is returned for reference — used by path-only overrides (the tutorial's public-weight re-evaluation).
        rec.update({"status": status if not overrides else "OK", "diffs": diffs, "max_abs_diff": worst,
                    "compared_with_overrides": bool(overrides)})
    else:
        rec["status"] = "OK"
    return rec


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="rsc-toolkit-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="outputs/<group>/<run> — directory containing best.pth (or --ckpt) and best.json")
    ap.add_argument("--project-root", default=None, help="base for relative paths (default: current directory)")
    ap.add_argument("--hydra-dir", default=None, help="config snapshot directory (default: auto-detect outputs/<date>/<time> with the same run.name)")
    ap.add_argument("--ckpt", default="best.pth", help="checkpoint file name inside run_dir")
    ap.add_argument("--override", action="append", default=[], help="OmegaConf dotlist, e.g. data.params.batch_size=16 (repeatable)")
    ap.add_argument("--sandbox", default=None, help="directory that receives all writes (default: <project-root>/.rsc_eval_sandbox; must be outside outputs/)")
    ap.add_argument("--tol", type=float, default=0.05, help="allowed |Δ| for PASS when comparing with best.json (metrics in %%)")
    ap.add_argument("--out", default=None, help="result JSON path (default: stdout only)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not args.quiet:
        logger.setLevel(logging.INFO)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rec = evaluate_run(pathlib.Path(args.run_dir), project_root=args.project_root, hydra_dir=args.hydra_dir,
                       ckpt_name=args.ckpt, overrides=args.override, sandbox=args.sandbox, tol=args.tol)
    text = json.dumps(rec, indent=1, ensure_ascii=False, default=str)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    m = rec.get("metrics") or {}
    summary = " ".join(f"{k}={m[k]:.2f}" for k in STD_KEYS if k in m)
    print(f"{rec['run']}: {rec.get('status')} {summary} (config {rec.get('config_source')})")
    if rec.get("status") not in ("PASS", "OK"):
        print(text, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
