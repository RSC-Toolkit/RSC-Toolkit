#!/usr/bin/env bash
# Generates notebooks/tutorial.ipynb and executes it with nbconvert. Default: Part A (toy, CPU) only, as in CI. --full also runs Part B (ICBHI, GPU).
#   scripts/tutorial/run_tutorial.sh            # RSC_TUTORIAL_QUICK=1
#   scripts/tutorial/run_tutorial.sh --full
# The executed notebook (with outputs) is written to $WORK/tutorial.executed.ipynb (WORK defaults to outputs/tutorial).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
# Interpreter: PY env var > .venv/bin/python > python/python3 on PATH  (|| true so a failing command -v does not silently exit under set -e)
if [ -z "${PY:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY=$(command -v python || command -v python3 || true); fi
fi
[ -n "$PY" ] || { echo "python not found — set PY=<interpreter>" >&2; exit 1; }
export RSC_TUTORIAL_QUICK=1
[ "${1:-}" = "--full" ] && export RSC_TUTORIAL_QUICK=0
WORK=${RSC_TUTORIAL_WORK:-$ROOT/outputs/tutorial}; mkdir -p "$WORK"; export RSC_TUTORIAL_WORK="$WORK"
"$PY" scripts/tutorial/make_notebook.py
"$PY" -m jupyter nbconvert --to notebook --execute notebooks/tutorial.ipynb \
  --ExecutePreprocessor.timeout=${TIMEOUT:-5400} --ExecutePreprocessor.kernel_name=python3 \
  --output-dir "$WORK" --output tutorial.executed.ipynb
echo "TUTORIAL-OK $WORK/tutorial.executed.ipynb (QUICK=$RSC_TUTORIAL_QUICK)"
