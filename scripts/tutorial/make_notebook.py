# -*- coding: utf-8 -*-
"""
Generator of notebooks/tutorial.ipynb. The notebook is built from this file, so edit the cells here and regenerate:

  .venv/bin/python scripts/tutorial/make_notebook.py            # → notebooks/tutorial.ipynb (clean notebook without outputs)
  scripts/tutorial/run_tutorial.sh [--full]                     # execute (Part A only by default; --full includes ICBHI)

Part A (toy, CPU, a few minutes) runs in CI every time; Part B (ICBHI download, 1 GPU epoch, public-weight re-evaluation) runs on a development machine.
"""
from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "tutorial.ipynb"   # alternative output path as an argument (for tests)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = []

cells.append(md(r"""
# RSC-Toolkit tutorial — from installation to training your own model

This notebook is the minimal path for **first-time users** of the toolkit, meant to be followed in one go.

| Part | What it does | Requirements | Time |
|---|---|---|---|
| A | train on the bundled toy data → re-evaluate → **register and train your own model** | CPU only | a few minutes |
| B | download the official ICBHI 2017 data → train 1 epoch → **re-evaluate a released paper weight** | 1 GPU, disk ≈ 8 GB | about 30 minutes, depending on download speed |

**Setup**: Python ≥ 3.9 (3.11 recommended); after cloning the repository, `pip install -e ".[trainer,models,notebook]"`
(for the same pins as the paper, `pip install -r requirements-lock.txt`). To plug in your own model with only the installed package,
see "Installing without a clone" in `tutorial/01-setup.md`.

Environment variables change the behavior: `RSC_TUTORIAL_QUICK=1` → skip Part B (as in CI). `RSC_TUTORIAL_WORK` → working directory (default `<clone>/outputs/tutorial`). `RSC_TUTORIAL_ROOT` → set the repository root explicitly.
"""))

cells.append(code(r"""
import json, os, pathlib, subprocess, sys

# Repository root (where pyproject.toml and rsc_toolkit/ are) — the notebook runs in notebooks/, so search upward through the parents
def find_root(start: pathlib.Path) -> pathlib.Path:
    env = os.environ.get("RSC_TUTORIAL_ROOT")
    if env:
        return pathlib.Path(env).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "rsc_toolkit").is_dir():
            return p
    raise SystemExit("repository root not found — run inside an RSC-Toolkit clone or set RSC_TUTORIAL_ROOT")

ROOT = find_root(pathlib.Path.cwd().resolve())
WORK = pathlib.Path(os.environ.get("RSC_TUTORIAL_WORK") or ROOT / "outputs" / "tutorial").resolve()
WORK.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("RSC_TUTORIAL_QUICK", "0") == "1"
import tempfile
SANDBOX = pathlib.Path(tempfile.mkdtemp(prefix="rsc_tutorial_sandbox_"))   # isolated write directory for re-evaluation — new on every run, outside outputs/ (enforced by the tool)

BIN = pathlib.Path(sys.executable).parent            # use the console scripts installed in this kernel's venv
RSCTK, TRAIN, EVAL = BIN / "rsctk", BIN / "rsc-toolkit-train", BIN / "rsc-toolkit-eval"
for exe in (RSCTK, TRAIN, EVAL):
    assert exe.exists(), f"{exe} not found — check that pip install -e '.[trainer,models,notebook]' was run in this kernel's environment"

ENV = {**os.environ,
       "RSCTK_PRETRAINED_ROOT": str(WORK / "pretrained"),          # where automatically downloaded pretrained weights are stored
       "HYDRA_FULL_ERROR": "1"}
# Cap BLAS/OpenCV threads per dataloader worker to 1 only for GPU training (Part B) — prevents a workers × thread-pool explosion on shared servers.
# CPU training (Part A) needs all cores, so it is not capped.
ENV_GPU = {**ENV, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1"}

def run(*cmd, cwd=WORK, check=True, env=None):
    # Show the command as is and run it (default cwd = WORK: training outputs outputs/... are created under WORK)
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), f"\n  (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env or ENV, text=True, capture_output=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
    print(tail)
    if check and proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}")
    return proc

import rsc_toolkit, torch
print("rsc_toolkit:", pathlib.Path(rsc_toolkit.__file__).parent, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("ROOT:", ROOT, "\nWORK:", WORK, "\nQUICK:", QUICK)
"""))

cells.append(md(r"""
## Part A — toy data (CPU)

`sample_data/toy_dataset/` is bundled synthetic data (no real recordings) in the same file format as the real ICBHI,
so index building → training → evaluation runs exactly as on the real data.

### A-1. Build the index
The toolkit never modifies the original audio; it only creates a **derived index** (`index/index_cycle.csv`). The train/test split is defined not by the index but by the data config's `split_file` (here `sample_data/toy_dataset/official_split.txt`).
"""))

cells.append(code(r"""
TOY = WORK / "toy"
run(RSCTK, "datasets", "build-index", "toy", "--project_root", ROOT, "--data_dir", TOY, "--sample_level", "cycle")
import pandas as pd
idx = pd.read_csv(TOY / "index" / "index_cycle.csv")
split = pd.read_csv(ROOT / "sample_data" / "toy_dataset" / "official_split.txt", sep=r"\s+", header=None, names=["recording", "split"])
print(idx.shape, "cycles;", idx["label"].value_counts().to_dict(), "| split file:", split["split"].value_counts().to_dict())
"""))

cells.append(md(r"""
### A-2. Training (CNN6, CPU, 1 epoch)

Every training run is a single line, `rsc-toolkit-train --config-name <config> key=value …`. Used here:
- `data=toy_fbank` — toy data + log-mel fbank features (cache in `data.params.features.cache_dir`)
- `model=cnn model.params.backbone=cnn6` — the toolkit's built-in PANNs-family CNN (6 layers)
- `callbacks=best_checkpoint` — saves `outputs/<run.group>/<run.name>/best.pth` and `best.json` when the validation Score is best
- `hydra.run.dir` — location of the config snapshot (`.hydra/config.yaml`). Re-evaluation rebuilds the model and data from this snapshot
"""))

cells.append(code(r"""
def train_toy(run_name: str, *extra, config_dir=None):
    # Hydra options (--config-dir, etc.) must come before the overrides (argparse does not accept interleaved positional arguments)
    opts = ["--config-dir", str(config_dir)] if config_dir else []
    model = ["model=cnn", "model.params.backbone=cnn6"] if not any(str(e).startswith("model=") for e in extra) else []
    run(TRAIN, "--config-name", "default", *opts, "data=toy_fbank", *model, "callbacks=best_checkpoint",
        f"data.params.project_root={ROOT}", f"data.params.data_dir={TOY}", f"data.params.features.cache_dir={WORK / 'cache_toy'}",
        "data.params.num_workers=0", "trainer.params.max_epochs=1", "trainer.params.accelerator=cpu",
        f"run.name={run_name}", "run.group=tutorial", f"hydra.run.dir={WORK / 'hydra' / run_name}",
        "+run.overwrite=true",   # re-running the notebook finds an existing best.pth for the same run name — explicitly allow overwriting
        *extra)
    best = json.loads((WORK / "outputs" / "tutorial" / run_name / "best.json").read_text())
    print("best.json:", best)
    return best

best_toy = train_toy("toy-cnn")
"""))

cells.append(md(r"""
### A-3. Re-evaluation

`rsc-toolkit-eval <run directory>` rebuilds the model and data from the snapshot with the same factories, strictly loads `best.pth`,
evaluates the validation split, and compares the result with `best.json` (PASS = |Δ| ≤ 0.05). Training outputs are only read; writes are isolated in `--sandbox` (a temporary directory outside outputs/).
"""))

cells.append(code(r"""
def evaluate(run_dir, hydra_dir=None, *overrides):
    out = WORK / f"{pathlib.Path(run_dir).name}.eval.json"
    cmd = [EVAL, run_dir, "--project-root", ROOT, "--sandbox", SANDBOX, "--out", out, "--quiet"]
    if hydra_dir is not None:
        cmd += ["--hydra-dir", hydra_dir]
    for ov in overrides:
        cmd += ["--override", ov]
    run(*cmd)
    res = json.loads(out.read_text())
    print(res["status"], {k: round(v, 3) for k, v in res["metrics"].items() if k in ("S_p", "S_e", "icbhi_score")}, "| |Δ| =", res.get("max_abs_diff"))
    return res

res_toy = evaluate(WORK / "outputs" / "tutorial" / "toy-cnn", WORK / "hydra" / "toy-cnn")
assert res_toy["status"] == "PASS", res_toy
"""))

cells.append(md(r"""
### A-4. Plug in your own model

No fork of the toolkit is needed. Write a class that inherits `BaseRSCModel` in a single file, implement `training_forward`, and register it;
the training, validation, and re-evaluation loops work unchanged (full contract: `rsc_toolkit/models/base.py`).

- The input `audio_batch` has the shape defined by the data config (`[B, F, T]` for fbank, no channel axis).
- `loss_func` is injected from the config's loss (weighted CE, etc.), so do not create one inside the model.
- Return `(logits [B, n_cls], loss)`.

Register the module with the environment variable `RSC_TOOLKIT_PLUGINS=<module>` (or the package entry point `rsc_toolkit.plugins`), and provide the model config
from a **directory outside the repository** with Hydra's `--config-dir`.
"""))

cells.append(code(r"""
PLUG = WORK / "plugins"; (PLUG / "configs" / "model").mkdir(parents=True, exist_ok=True)
(PLUG / "my_model.py").write_text('''
import torch
from torch import nn
from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import ModelFactory


@ModelFactory.register_model_class("my_model")
class MyModel(BaseRSCModel, nn.Module):
    # Small 2-layer CNN. n_cls is injected via model.params.n_cls.

    def __init__(self, n_cls: int = 4, hidden: int = 16, **params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                 # [B, F, T] → [B, 1, F, T]
            x = x.unsqueeze(1)
        return self.net(x)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
''')
(PLUG / "configs" / "model" / "my_model.yaml").write_text('''
name: my_model
params:
  n_cls: 4
  hidden: 16
''')
ENV["RSC_TOOLKIT_PLUGINS"] = "my_model"                       # module name the toolkit imports
ENV["PYTHONPATH"] = str(PLUG) + os.pathsep + ENV.get("PYTHONPATH", "")

best_my = train_toy("toy-mymodel", "model=my_model", config_dir=PLUG / "configs")
res_my = evaluate(WORK / "outputs" / "tutorial" / "toy-mymodel", WORK / "hydra" / "toy-mymodel")
assert res_my["status"] == "PASS", res_my
"""))

cells.append(md(r"""
End of Part A. Changing only `data=`, `model=`, and `--config-name` in the same commands runs the paper reproduction recipes (the table in the README).

## Part B — ICBHI 2017 real data (GPU)

With `RSC_TUTORIAL_QUICK=1` the cells below are skipped.

### B-1. Download · index · validate
Downloads the official release zip (1.98 GB, 920 recordings), extracts it into `WORK/icbhi/`, and builds the cycle index with the official 60/40 split.
"""))

cells.append(code(r"""
ICBHI = WORK / "icbhi"
if QUICK:
    print("skipped (RSC_TUTORIAL_QUICK=1)")
else:
    run(RSCTK, "datasets", "download", "icbhi", "--project_root", ROOT, "--data_dir", ICBHI)        # skipped if already present
    run(RSCTK, "datasets", "build-index", "icbhi", "--project_root", ROOT, "--data_dir", ICBHI, "--sample_level", "cycle")
    run(RSCTK, "datasets", "validate", "icbhi", "--project_root", ROOT, "--data_dir", ICBHI)
    idx = pd.read_csv(ICBHI / "index" / "index_cycle.csv")
    print(idx.shape, "cycles;", idx["label"].value_counts().sort_index().to_dict())
"""))

cells.append(md(r"""
### B-2. 1-epoch training (CNN6 fine-tuning recipe)

The paper reproduction config `sgscl_nonast_icbhi` + `model.name=cnn6_sft` is the "CNN6 fine-tuning" row of Table 1 (400 epochs in the full recipe).
Here only 1 epoch runs to check the pipeline. The PANNs Cnn6 pretrained weights are downloaded automatically from zenodo (record 3960586) to `$RSCTK_PRETRAINED_ROOT/upstream/Cnn6_mAP=0.343.pth` — on networks where zenodo is blocked, place that file at the same path manually. This is GPU training, so the per-worker thread cap (`ENV_GPU`) is applied.
"""))

cells.append(code(r"""
if QUICK:
    print("skipped (RSC_TUTORIAL_QUICK=1)")
else:
    name = "icbhi-cnn6-1epoch"
    run(TRAIN, "--config-name", "sgscl_nonast_icbhi", "model.name=cnn6_sft",
        f"data.params.project_root={ROOT}", f"data.params.data_dir={ICBHI}", f"data.params.features.cache_dir={WORK / 'cache_icbhi'}",
        "trainer.params.max_epochs=1", f"run.name={name}", "run.group=tutorial", f"hydra.run.dir={WORK / 'hydra' / name}",
        "+run.overwrite=true", env=ENV_GPU)
    # the sgscl_nonast_paper callbacks save to outputs/sgscl/<run.name>
    print(json.loads((WORK / "outputs" / "sgscl" / name / "best.json").read_text()))
    res_1ep = evaluate(WORK / "outputs" / "sgscl" / name, WORK / "hydra" / name)
    assert res_1ep["status"] == "PASS", res_1ep
"""))

cells.append(md(r"""
### B-3. Re-evaluating a released paper weight

The weights behind the paper tables are released as run bundles (`best.pth` · `best.json` · `.hydra/config.yaml`). The list and links are
in `weights_manifest.csv`; `scripts/download_paper_weights.py` (gdown, included in the `notebook` extra) downloads them and checks sha256.
If the link (a Drive folder link: the run folder itself or a parent folder holding run folders) is still the `[Google Drive]` placeholder, only a notice is printed — in that case place the bundle in `WORK/paper_weights/<run>/` manually.

A downloaded bundle contains its config snapshot, so `rsc-toolkit-eval <bundle> --override data.params.data_dir=<your ICBHI path>`
re-evaluates it directly. An evaluation with a path override returns `status: OK` together with the |Δ| against `best.json` (the value in the paper table); since the official test set is the same, |Δ| is 0.
"""))

cells.append(code(r"""
if QUICK:
    print("skipped (RSC_TUTORIAL_QUICK=1)")
else:
    RUN = "icbhi-cnn6-ce-s1"                                   # Table 1 "CNN6 fine-tuning" seed 1 (17 MB)
    bundle = WORK / "paper_weights" / RUN
    if not (bundle / "best.pth").exists():
        proc = run(sys.executable, ROOT / "scripts" / "download_paper_weights.py", "--run", RUN, "--out", WORK / "paper_weights", check=False)
        if proc.returncode == 2:
            print(f"\nNo public link yet. Place the bundle in {bundle} and run this cell again.")
    if (bundle / "best.pth").exists():
        res_paper = evaluate(bundle, None, f"data.params.data_dir={ICBHI}", f"data.params.features.cache_dir={WORK / 'cache_icbhi'}")
        print("paper table values (best.json):", json.loads((bundle / "best.json").read_text()))
        # With a path override the tool returns status=OK and the comparison (|Δ|) instead of a verdict — the official test set is the same, so |Δ| must be 0
        worst = res_paper.get("max_abs_diff")
        assert res_paper["status"] in ("PASS", "OK") and worst is not None and worst <= 0.05, res_paper   # 0.0 is a valid value, so do not replace this with `or`
"""))

cells.append(md(r"""
## Next
- Reproduction recipes for the 29 papers (`--config-name` and overrides per model × dataset) are in the table of the README section "Reproducing a method".
- To distribute your own model as a package, declare `[project.entry-points."rsc_toolkit.plugins"]` in `pyproject.toml` (README "Adding your own model").
- Losses, metrics, datamodules, and processors use the same registry mechanism (`rsc_toolkit/registries.py`).
"""))

for i, c in enumerate(cells):                       # deterministic cell ids — regeneration is byte-identical to the committed file (random ids would dirty git)
    c["id"] = f"cell-{i:02d}"
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
