#!/bin/bash
# 학습된 weight(outputs/<계열>/<run>/best.pth) 를 불러 test 를 다시 돌린다 — rsc-toolkit-eval 의 얇은 셸 래퍼.
#
#   scripts/eval.sh <run-dir> [rsc-toolkit-eval 옵션...]
#
#   예) scripts/eval.sh outputs/patchmix/icbhi-patchmix-ce-s1                     # best.json 과 자동 대조 (PASS/MISMATCH)
#       GPU=2 scripts/eval.sh outputs/qlung/icbhi-qlung-s1 --override data.params.batch_size=16 --out eval.json
#       scripts/eval.sh outputs/m2d/icbhi-m2dx-s5-s1 --hydra-dir outputs/2026-08-27/10-12-33     # 스냅샷 직접 지정
#
# 설정은 그 run 의 hydra 스냅샷(run.name 이 같은 outputs/<날짜>/<시각>/.hydra)을 자동으로 찾아 학습 때와 같은 팩토리로
# 모델·데이터를 다시 세우고 strict 로드 → validate. 학습 산출물은 읽기만 하고(쓰기는 샌드박스), best.json 이 있으면 대조한다.
# 여러 run 전수 검증은 scripts/verify_paper_weights.sh (GPU 병렬·요약표).
#
# 환경변수: GPU(기본 0) · PY(기본 .venv/bin/python) · SANDBOX(기본 /tmp/rsc_eval_sandbox_<uid>) · THREAD_CAP(기본 1)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
GPU=${GPU:-0}
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
