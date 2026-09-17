# RSC-Toolkit tutorial (minimal path)

This is the shell version of [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb).
Requirements: Python ≥ 3.9 (3.11 recommended). Part A runs on a CPU in a few minutes. Part B needs one GPU and about 8 GB of
disk (ICBHI zip 1.98 GB + extracted 3.9 GB + feature cache 2.7 GB) and takes roughly 30 minutes depending on download speed.

## 0. Installation

```bash
git clone https://github.com/RSC-Toolkit/RSC-Toolkit.git RSC-Toolkit && cd RSC-Toolkit
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[trainer,models,notebook]"          # exact pins: pip install -r requirements-lock.txt, then pip install -e . --no-deps
# CPU only (Part A): install the CPU build of torch first to download about 1 GB instead of 6 GB
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
python -m rsc_toolkit.cli --help                     # or: rsctk --help
```

Pick a working directory. The commands below use `WORK=$PWD/outputs/tutorial` (ignored by git). Training outputs are
written to `outputs/<group>/<run>/` under the current directory, so run the commands from `WORK`.

```bash
export WORK=$PWD/outputs/tutorial ROOT=$PWD; mkdir -p "$WORK"; cd "$WORK"
export RSCTK_PRETRAINED_ROOT=$WORK/pretrained                                                # where downloaded backbone weights go
SANDBOX=$(mktemp -d /tmp/rsc_tutorial_sandbox.XXXXXX)                                        # write-isolation directory for re-evaluation (outside outputs/)
```

## Part A — toy data (CPU)

### A-1. Index
```bash
rsctk datasets build-index toy --project_root "$ROOT" --data_dir "$WORK/toy" --sample_level cycle
```

### A-2. Training (CNN6, 1 epoch)
```bash
rsc-toolkit-train --config-name default data=toy_fbank model=cnn model.params.backbone=cnn6 callbacks=best_checkpoint \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/toy" data.params.features.cache_dir="$WORK/cache_toy" \
  data.params.num_workers=0 trainer.params.max_epochs=1 trainer.params.accelerator=cpu \
  run.name=toy-cnn run.group=tutorial hydra.run.dir="$WORK/hydra/toy-cnn" +run.overwrite=true
cat "$WORK/outputs/tutorial/toy-cnn/best.json"
```
`callbacks=best_checkpoint` writes `best.pth` and `best.json` for the epoch with the best validation Score to
`outputs/<run.group>/<run.name>/`; `.hydra/config.yaml` under `hydra.run.dir` is the config snapshot.

> `+run.overwrite=true`: if `best.pth`/`best.json` already exist for the same `run.name`, the toolkit stops instead of
> overwriting them. The tutorial is meant to be re-run, so it allows overwriting explicitly.

### A-3. Re-evaluation
```bash
rsc-toolkit-eval "$WORK/outputs/tutorial/toy-cnn" --project-root "$ROOT" --hydra-dir "$WORK/hydra/toy-cnn" --sandbox "$SANDBOX"
```
The snapshot is used to rebuild the model and data through the same factories, `best.pth` is loaded strictly and evaluated,
and the result is compared with `best.json` (PASS when |Δ| ≤ 0.05). Pass the run directory as an absolute path (relative
paths are resolved against `--project-root`). `--sandbox` isolates all writes and must be outside `outputs/`.

### A-4. Your own model
One file is enough. Subclass `BaseRSCModel`, implement `training_forward`, and register the class (contract:
`rsc_toolkit/models/base.py`).

```bash
mkdir -p "$WORK/plugins/configs/model"
cat > "$WORK/plugins/my_model.py" <<'PY'
import torch
from torch import nn
from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import ModelFactory

@ModelFactory.register_model_class("tutorial_cnn")
class TutorialCNN(BaseRSCModel, nn.Module):
    def __init__(self, n_cls: int = 4, hidden: int = 16, **params):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(), nn.MaxPool2d(4),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(hidden, n_cls))
    def forward(self, x):
        return self.net(x.unsqueeze(1) if x.dim() == 3 else x)      # [B, F, T] -> [B, 1, F, T]
    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
PY
printf 'name: tutorial_cnn\nparams:\n  n_cls: 4\n  hidden: 16\n' > "$WORK/plugins/configs/model/tutorial_cnn.yaml"

RSC_TOOLKIT_PLUGINS=my_model PYTHONPATH="$WORK/plugins" \
rsc-toolkit-train --config-name default --config-dir "$WORK/plugins/configs" data=toy_fbank model=tutorial_cnn callbacks=best_checkpoint \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/toy" data.params.features.cache_dir="$WORK/cache_toy" \
  data.params.num_workers=0 trainer.params.max_epochs=1 trainer.params.accelerator=cpu \
  run.name=toy-mymodel run.group=tutorial hydra.run.dir="$WORK/hydra/toy-mymodel" +run.overwrite=true
RSC_TOOLKIT_PLUGINS=my_model PYTHONPATH="$WORK/plugins" \
rsc-toolkit-eval "$WORK/outputs/tutorial/toy-mymodel" --project-root "$ROOT" --hydra-dir "$WORK/hydra/toy-mymodel" --sandbox "$SANDBOX"
```
- `audio_batch` has the shape defined by the data config (`[B, F, T]` for fbank features), and `loss_func` is the loss
  from the config. Return `(logits [B, n_cls], loss)`.
- Registration: the environment variable `RSC_TOOLKIT_PLUGINS=<module>` or the package entry point `rsc_toolkit.plugins`.
  The model config is supplied from outside the repository with Hydra's `--config-dir`.
- Losses, metrics, datamodules and processors use the same registry mechanism — see "Adding your own model" in the README.

## Part B — ICBHI 2017 (GPU)

### B-1. Download, index, validate
```bash
rsctk datasets download    icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi"       # official zip, 1.98 GB
rsctk datasets build-index icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi" --sample_level cycle
rsctk datasets validate    icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi"
```

### B-2. One epoch of training (CNN6 fine-tuning recipe; the full recipe uses 400 epochs)
For GPU training, cap BLAS/OpenCV threads per dataloader worker at 1 (this prevents worker × thread-pool oversubscription on
shared servers; Part A trains on the CPU and does not need it).
The PANNs Cnn6 pre-trained weights are downloaded automatically from Zenodo (record 3960586) to
`$RSCTK_PRETRAINED_ROOT/upstream/Cnn6_mAP=0.343.pth`. If Zenodo is blocked on your network, place that file at the same path.
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
rsc-toolkit-train --config-name sgscl_nonast_icbhi model.name=cnn6_sft \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/icbhi" data.params.features.cache_dir="$WORK/cache_icbhi" \
  trainer.params.max_epochs=1 run.name=icbhi-cnn6-1epoch run.group=tutorial hydra.run.dir="$WORK/hydra/icbhi-cnn6-1epoch" +run.overwrite=true
rsc-toolkit-eval "$WORK/outputs/sgscl/icbhi-cnn6-1epoch" --project-root "$ROOT" --hydra-dir "$WORK/hydra/icbhi-cnn6-1epoch" --sandbox "$SANDBOX"
```
(The callbacks of this recipe save to `$WORK/outputs/sgscl/<run.name>`.)

### B-3. Re-evaluating a released paper weight
The weights behind the benchmark tables are released as run bundles (`best.pth`, `best.json`, `.hydra/config.yaml`).
[`weights_manifest.csv`](weights_manifest.csv) lists them with download links; `scripts/download_paper_weights.py` downloads
a bundle (uses gdown, included in the `notebook` extra) and verifies its sha256 checksums. Links are Google Drive folder
links (`…/folders/<id>`), either the run folder itself or a parent folder that contains run folders.

```bash
python "$ROOT/scripts/download_paper_weights.py" --list
python "$ROOT/scripts/download_paper_weights.py" --run icbhi-cnn6-ce-s1 --out "$WORK/paper_weights"
rsc-toolkit-eval "$WORK/paper_weights/icbhi-cnn6-ce-s1" --project-root "$ROOT" --sandbox "$SANDBOX" \
  --override data.params.data_dir="$WORK/icbhi" --override data.params.features.cache_dir="$WORK/cache_icbhi"
```
The bundle carries its own config snapshot, so it is re-evaluated without an `outputs/` directory. When paths are
overridden, the evaluation reports `status: OK` together with |Δ| against `best.json` (the released value); on the same
official test set |Δ| is 0 (without overrides it reports PASS/MISMATCH).

## Using the installed package only
If you only want to plug in your own model, a clone is not required:
`pip install "rsc-toolkit[trainer,models] @ git+https://github.com/RSC-Toolkit/RSC-Toolkit.git"`.
The same commands work, but the toy data (`sample_data/`) is not included in the wheel, so start from your own data or
ICBHI (Part B) instead of Part A.

## Tools at a glance
| task | command |
|---|---|
| train a method recipe | `scripts/train.sh <config> [override…]` (see the README table) |
| re-evaluate one run | `rsc-toolkit-eval outputs/<group>/<run>` |
| run this tutorial automatically | `scripts/tutorial/run_tutorial.sh [--full]` (CI runs Part A) |
