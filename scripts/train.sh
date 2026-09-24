#!/bin/bash
# Runs one training run with the selected config (= paper recipe) — a thin shell wrapper around rsc-toolkit-train.
#
#   scripts/train.sh <config-name> [hydra overrides...]
#
#   e.g. scripts/train.sh patchmix_icbhi seed=1 run.name=my-patchmix-s1
#        GPU=3 scripts/train.sh sgscl_nonast_icbhi model.name=resnet18_sft data=icbhi_sgscl_nonast seed=2
#        scripts/train.sh m2d_icbhi trainer.params.max_epochs=1 ++trainer.params.limit_train_batches=2   # smoke test
#
# Model and data are selected with config group overrides: model=<configs/model/*.yaml name> data=<configs/data/*.yaml name>
# (or model.name=<registered name>). Registered names: rsc-toolkit list (or python -c "import rsc_toolkit.registries as r; print(sorted(r.MODELS))").
# Output: outputs/<family>/<run.name>/best.pth and best.json (family = the config's callbacks dirpath), hydra snapshot outputs/<date>/<time>/.hydra.
#
# Environment variables: GPU (default: CUDA_VISIBLE_DEVICES from the environment, else 0) · PY (default .venv/bin/python) · THREAD_CAP (default 1 — prevents
#           OpenCV/OpenBLAS thread explosion in dataloader workers when several runs share a machine; 0 disables the cap)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
GPU=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}   # if GPU is unset, respect CUDA_VISIBLE_DEVICES from the environment
if [ $# -lt 1 ]; then
  sed -n 2,16p "$0"; exit 1
fi
CONFIG="$1"; shift
if [ "${THREAD_CAP:-1}" != "0" ]; then
  export OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fi
cd "$ROOT"
echo "train: config=$CONFIG gpu=$GPU overrides=[$*]"
CUDA_VISIBLE_DEVICES=$GPU exec "$PY" -m rsc_toolkit.train --config-name "$CONFIG" "$@"
