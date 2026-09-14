#!/usr/bin/env bash
# notebooks/tutorial.ipynb 를 생성하고 nbconvert 로 실행한다. 기본은 Part A(toy, CPU) 만 — CI 가 이렇게 돈다. --full 은 Part B(ICBHI, GPU)까지.
#   scripts/tutorial/run_tutorial.sh            # RSC_TUTORIAL_QUICK=1
#   scripts/tutorial/run_tutorial.sh --full
# 실행된 노트북(출력 포함)은 $WORK/tutorial.executed.ipynb 에 남는다 (WORK 기본 outputs/tutorial).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
# 인터프리터: PY 환경변수 > .venv/bin/python > PATH 의 python/python3  (set -e 아래에서 command -v 실패가 조용히 죽이지 않게 || true)
if [ -z "${PY:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY=$(command -v python || command -v python3 || true); fi
fi
[ -n "$PY" ] || { echo "python 을 찾지 못했다 — PY=<interpreter> 로 지정하라" >&2; exit 1; }
export RSC_TUTORIAL_QUICK=1
[ "${1:-}" = "--full" ] && export RSC_TUTORIAL_QUICK=0
WORK=${RSC_TUTORIAL_WORK:-$ROOT/outputs/tutorial}; mkdir -p "$WORK"; export RSC_TUTORIAL_WORK="$WORK"
"$PY" scripts/tutorial/make_notebook.py
"$PY" -m jupyter nbconvert --to notebook --execute notebooks/tutorial.ipynb \
  --ExecutePreprocessor.timeout=${TIMEOUT:-5400} --ExecutePreprocessor.kernel_name=python3 \
  --output-dir "$WORK" --output tutorial.executed.ipynb
echo "TUTORIAL-OK $WORK/tutorial.executed.ipynb (QUICK=$RSC_TUTORIAL_QUICK)"
