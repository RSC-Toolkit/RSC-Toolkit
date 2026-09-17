#!/bin/bash
# Loads trained weights (outputs/<family>/<run>/best.pth) and re-runs the test set — a thin shell wrapper around rsc-toolkit-eval.
#
#   scripts/eval.sh <run-dir> [rsc-toolkit-eval options...]
#
#   e.g. scripts/eval.sh outputs/patchmix/icbhi-patchmix-ce-s1                   # compared against best.json automatically (PASS/MISMATCH)
#        GPU=2 scripts/eval.sh outputs/qlung/icbhi-qlung-s1 --override data.params.batch_size=16 --out eval.json
#        scripts/eval.sh outputs/m2d/icbhi-m2dx-s5-s1 --hydra-dir outputs/2026-08-27/10-12-33     # explicit snapshot
#
# The config comes from the run's hydra snapshot (outputs/<date>/<time>/.hydra with the same run.name), found automatically; the model
# and data are rebuilt with the same factories as in training, then strict load -> validate. Training outputs are read only (writes go to the sandbox); best.json is compared if present.
#
# Environment variables: GPU (default: CUDA_VISIBLE_DEVICES from the environment, else 0) · PY (default .venv/bin/python) · SANDBOX (default /tmp/rsc_eval_sandbox_<uid>) · THREAD_CAP (default 1)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
GPU=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}   # if GPU is unset, respect CUDA_VISIBLE_DEVICES from the environment
SANDBOX=${SANDBOX:-${TMPDIR:-/tmp}/rsc_eval_sandbox_$(id -u)}
if [ $# -lt 1 ]; then
  sed -n 2,14p "$0"; exit 1
fi
RUN_DIR="$1"; shift
if [ "${THREAD_CAP:-1}" != "0" ]; then
  export OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fi
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false
cd "$ROOT"
echo "eval: run_dir=$RUN_DIR gpu=$GPU sandbox=$SANDBOX args=[$*]"
CUDA_VISIBLE_DEVICES=$GPU exec "$PY" -m rsc_toolkit.evaluate "$RUN_DIR" --project-root "$ROOT" --sandbox "$SANDBOX" "$@"
