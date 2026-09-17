"""체크포인트 재평가 — `rsc-toolkit-eval` (리팩토링 P2-3).

학습 run 디렉터리(`outputs/<group>/<run>/best.pth`)를 그 run 의 설정 스냅샷(`outputs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml`,
없으면 run 로그에 찍힌 resolved config)으로 **학습 때와 같은 팩토리**로 다시 세우고, state_dict 를 strict 로 로드한 뒤
검증 스플릿을 `trainer.validate` 로 평가해 지표를 돌려준다.

원칙
- 학습 산출물은 **읽기만 한다**: 프로세스 CWD 를 샌드박스로 옮기고(입력 디렉터리만 심링크), 체크포인트·json 을 쓰는
  콜백은 버리거나 샌드박스로 돌리며, 로거는 끈다. 피처 캐시의 체인 대장(`_chain.json`)도 읽기만 한다.
- 설정은 스냅샷 그대로 쓰되 `--override k=v`(OmegaConf dotlist) 로 덮어쓸 수 있다 — 예: 다른 배치 크기, 다른 split.

    rsc-toolkit-eval outputs/patchmix/icbhi-patchmix-ce-s1                 # best.json 과 자동 대조
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

# 입력으로만 쓰는 최상위 디렉터리 — 샌드박스에 심링크로 노출한다. outputs/ 는 일부러 뺀다.
READ_ONLY_ROOTS: Tuple[str, ...] = ("rsc_toolkit", ".datasets", "downloaded_datasets", "pretrained_models", "sample_data")
# config 안의 상대 경로를 절대 경로로 바꿀 접두사 (outputs/ 는 encoder_ckpt 같은 **입력** 에만 등장한다 —
# 콜백 dirpath 는 sanitize_config 가 따로 처리한다).
ABS_PREFIXES: Tuple[str, ...] = tuple(f"{r}/" for r in READ_ONLY_ROOTS) + ("outputs/",)
# validate 에서 살려 두는 콜백 — 지표·손실 재현에 필요하거나 무해한 것만. 나머지(체크포인트·json 저장, MA/EMA,
# LR 스케줄, 샘플러 epoch 등)는 전부 버린다.
KEEP_CALLBACKS = frozenset({"AMSDEpochCallback", "DomainTransferStages", "ProtoNetClassWeight", "RespireNetVoteEval"})
STD_KEYS: Tuple[str, ...] = ("S_p", "S_e", "icbhi_score")
VOTE_MAP = {"S_p": "vote/S_p", "S_e": "vote/S_e", "icbhi_score": "vote/icbhi_score", "cycle_acc": "vote/cycle_acc"}


# ----------------------------------------------------------------------------- 설정 복구
def load_snapshot_config(hydra_dir: pathlib.Path, project_root: pathlib.Path) -> dict:
    """`<hydra_dir>/.hydra/config.yaml` 을 읽는다. `${hydra:runtime.cwd}` 는 hydra 런타임 밖에서 못 푸므로 project_root 로 치환."""
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
    """train.py 가 시작 직후 찍는 `OmegaConf.to_yaml(cfg, resolve=True)` 블록을 파싱한다 (마지막 블록)."""
    import yaml
    from omegaconf import OmegaConf

    lines = pathlib.Path(log_path).read_text(errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln == "model:"]
    if not starts:
        return None
    block = []
    for ln in lines[starts[-1]:]:
        # 최상위 키(소문자 key:) 또는 들여쓰기 줄만 YAML 블록에 속한다. 로그 줄·경고·Lightning 출력에서 멈춘다.
        if ln.startswith((" ", "-")) or re.match(r"^[a-z_]+:( .*)?$", ln):
            block.append(ln)
        else:
            break
    cfg = yaml.safe_load("\n".join(block))
    if not isinstance(cfg, dict) or "model" not in cfg or "data" not in cfg:
        return None
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def _snapshot_run_name(config_yaml: pathlib.Path) -> Optional[str]:
    """스냅샷의 run.name (해석 없이). 기본값 `${data.name}-${model.name}` 이면 두 이름으로 조립한다."""
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
    """`outputs/*/*/.hydra/config.yaml` 중 run.name 이 같은 스냅샷 디렉터리들 (경로 사전순)."""
    outputs = pathlib.Path(project_root) / "outputs"
    found = []
    for cfg_yaml in sorted(outputs.glob("*/*/.hydra/config.yaml")):
        if _snapshot_run_name(cfg_yaml) == run_name:
            found.append(cfg_yaml.parent.parent)
    return found


def pick_snapshot(snapshot_dirs: Sequence[pathlib.Path], best_pth: pathlib.Path) -> Optional[pathlib.Path]:
    """같은 run.name 의 스냅샷이 여럿이면 **best.pth 가 저장되기 직전** 것(디렉터리명 = 시작 시각). 없으면 마지막 것."""
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
    """run 디렉터리의 설정을 복구한다: 지정 hydra_dir → run_dir/.hydra → run.name 이 같은 스냅샷 → run 로그. (cfg, 출처)"""
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
    """상대 경로 문자열(READ_ONLY_ROOTS·outputs 접두사)을 project_root 기준 절대 경로로. skip_keys 서브트리는 건너뛴다."""
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
    """평가 전용 설정으로 바꾼다: resume 제거, 경로 절대화, 쓰기 콜백 제거(유지 목록의 dirpath 는 샌드박스로), 로거·체크포인팅 끔."""
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
            if "dirpath" in cb:  # RespireNetVoteEval — best.pth/best.json 을 쓰므로 샌드박스로
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
    """`k.sub=v` dotlist 를 OmegaConf 로 병합한다 (값은 YAML 로 해석: 8 → int, true → bool)."""
    if not overrides:
        return cfg
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(merged, resolve=True)


# ----------------------------------------------------------------------------- 샌드박스·캐시 대장
def make_sandbox(path: pathlib.Path, project_root: pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in READ_ONLY_ROOTS:
        src, dst = pathlib.Path(project_root) / name, path / name
        if not src.exists():
            continue
        if dst.is_symlink() and dst.resolve() != src.resolve():
            dst.unlink()                      # 다른 checkout 이 남긴 샌드박스를 재사용할 때 조용히 그쪽을 읽지 않도록 다시 잇는다
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    (path / "lightning").mkdir(exist_ok=True)
    (path / "callback_out").mkdir(exist_ok=True)


def install_readonly_chain_guard(notes: list):
    """FeatureCache._check_audio_chain 을 읽기 전용으로 바꾼다. 되돌리는 함수를 돌려준다.

    원본은 대장(_chain.json)이 없으면 **실제 캐시 디렉터리에 새로 쓰고**, 어긋나면 예외를 던진다.
    평가는 캐시를 읽기만 해야 하므로 쓰기를 없애고, 어긋남은 예외 대신 `notes` 에 기록한다
    (waveform 종류는 캐시를 아예 안 쓰므로 어긋나도 평가에 영향이 없다 — 그 사실도 기록에 담긴다)."""
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
    """`install_readonly_chain_guard` 의 with-블록 형태 — 블록을 나가면 원래 동작으로 되돌린다."""
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


# ----------------------------------------------------------------------------- 로드·평가
# 학습 샘플 수에 따라 크기가 정해지는 버퍼(AMS-D 의 per-sample 상태) — 이 이름들만 ckpt 모양으로 맞춘다.
# 그 밖의 모양 불일치(BN running stat·pos_embed 등)는 모델 구성이 다르다는 뜻이므로 strict 로드가 실패해야 한다
# (예전엔 이름 불문 전부 0 으로 바꿔 "strict_ok" 로 통과시켰다 — REVIEW 2026-09-10 §4 P3).
ADAPTABLE_BUFFER_SUFFIXES = ("all_predictions", "ids_masked")


def adapt_buffer_shapes(model, sd: dict) -> List[str]:
    """ckpt 의 버퍼 모양이 새 모델과 다르면(AMSD 의 per-sample 버퍼) 그 버퍼만 ckpt 모양으로 바꾼다."""
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
    """best.pth(`{"model": state_dict}` 또는 state_dict 자체)를 strict 로 로드한다. 결과는 info dict (load: strict_ok|strict_fail)."""
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
            # 모양 불일치(버퍼·가중치)는 strict=False 로도 못 싣는다 — 부분 로드 정보 없이 실패로 남긴다
            info["load_error"] = str(exc2)[:500]
    return info


def build_and_validate(cfg: dict, ckpt_path: pathlib.Path, sandbox: pathlib.Path, run_name: str) -> dict:
    """설정(sanitize 된 것)으로 모델·손실·지표·datamodule·trainer 를 세우고 ckpt 를 로드해 validate. 반환: info(load, metrics…)."""
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

    # Lightning 은 validate 에 setup("validate") 를 부르는데 툴킷 datamodule 들은 fit/test 만 안다 → None(전부) 로 매핑.
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
    """best.json(expected) 과 재평가 지표를 대조 — PASS(|Δ| ≤ tol) · MISMATCH · ERROR(대조할 키 없음)."""
    if "cycle_acc" in expected:  # RespireNet — 사이클 투표 지표
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
    """run 디렉터리 하나를 재평가한다. 반환 dict: run, config_source, dropped_callbacks, load, metrics, (best.json 있으면) status·diffs."""
    project_root = pathlib.Path(project_root or os.getcwd()).resolve()
    run_dir = pathlib.Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = (project_root / run_dir).resolve()
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트가 없다: {ckpt_path}")
    rec: Dict[str, Any] = {"run": run_dir.name, "run_dir": str(run_dir), "ckpt": str(ckpt_path)}

    cfg, source = resolve_run_config(run_dir, project_root, hydra_dir=hydra_dir, ckpt_name=ckpt_name)
    rec["config_source"] = source
    if cfg is None:
        rec["status"] = "SKIP"
        rec["note"] = "설정 스냅샷도 run 로그도 없다 — --hydra-dir 로 지정하라"
        return rec

    sandbox = pathlib.Path(sandbox) if sandbox else pathlib.Path(project_root) / ".rsc_eval_sandbox"
    if (project_root / "outputs") in sandbox.resolve().parents:
        raise ValueError(f"샌드박스 {sandbox} 가 outputs/ 안이다 — 콜백 산출물이 결과 디렉터리에 섞인다")
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
        # override 가 있으면 프로토콜이 바뀌었을 수 있으므로 판정(PASS/MISMATCH)은 내리지 않고 "OK" 로 두되,
        # 대조 결과(diffs·max_abs_diff)는 참고용으로 함께 돌려준다 — 경로만 바꾼 평가(튜토리얼의 공개 weight 재평가)가 이를 쓴다.
        rec.update({"status": status if not overrides else "OK", "diffs": diffs, "max_abs_diff": worst,
                    "compared_with_overrides": bool(overrides)})
    else:
        rec["status"] = "OK"
    return rec


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="rsc-toolkit-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="outputs/<group>/<run> — best.pth(또는 --ckpt) 와 best.json 이 있는 디렉터리")
    ap.add_argument("--project-root", default=None, help="상대 경로의 기준 (기본: 현재 디렉터리)")
    ap.add_argument("--hydra-dir", default=None, help="설정 스냅샷 디렉터리 (기본: run.name 이 같은 outputs/<날짜>/<시각> 자동 탐색)")
    ap.add_argument("--ckpt", default="best.pth", help="run_dir 안의 체크포인트 파일명")
    ap.add_argument("--override", action="append", default=[], help="OmegaConf dotlist, 예: data.params.batch_size=16 (반복 가능)")
    ap.add_argument("--sandbox", default=None, help="쓰기가 떨어질 디렉터리 (기본: <project-root>/.rsc_eval_sandbox; outputs/ 밖이어야 한다)")
    ap.add_argument("--tol", type=float, default=0.05, help="best.json 대조 PASS 허용 |Δ| (%% 단위 지표)")
    ap.add_argument("--out", default=None, help="결과 JSON 경로 (기본: 표준출력만)")
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
