#!/bin/bash
# 논문 weight 파일 전수 검증 — GPU 병렬 래퍼.
#
# 각 run 의 best.pth 를 그 run 의 설정 스냅샷으로 재구성한 모델에 strict 로드하고, 공식 test 스플릿
# (ICBHI 60/40 · SPRSound 2022 inter-patient) 을 재평가해 best.json 의 S_p/S_e/Score 와 대조한다.
# outputs/ 는 읽기만 한다 — 모든 쓰기는 $OUT/sandbox 로 격리된다 (scripts/verify_paper_weights.py 참조).
#
# 사용:
#   scripts/verify_paper_weights.sh                       # results_master.csv 전체, 가용 GPU 전부 병렬
#   scripts/verify_paper_weights.sh --dataset icbhi       # 데이터셋 한정 (icbhi | sprsound)
#   scripts/verify_paper_weights.sh --group patchmix --group sgscl
#   scripts/verify_paper_weights.sh --smoke               # 그룹·출처별 1개씩 (빠른 점검)
#   GPUS=0,1,2 OUT=outputs/verify_2026-09-06 scripts/verify_paper_weights.sh ...
#
# 환경변수: GPUS(쉼표 구분, 기본 nvidia-smi 의 전부) · OUT(기본 outputs/verify_<날짜>) · TOL(기본 0.05)
#           SANDBOX(기본 /tmp/rsc_verify_sandbox_<epoch> — 반드시 outputs/ 밖)
# 결과: $OUT/results.csv, $OUT/summary.md (PASS / MISMATCH / LOAD_FAIL / ERROR / SKIP)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/verify_$(date +%F)}
TOL=${TOL:-0.05}
# 샌드박스(모든 쓰기가 떨어지는 CWD)는 outputs/ 밖에 둔다 — RespireNet 콜백이 쓰는 best.json 이
# outputs/ 안에 생기면 build_results_master.py 가 논문 표에 주워 담는다.
SANDBOX=${SANDBOX:-${TMPDIR:-/tmp}/rsc_verify_sandbox_$(date +%s)}
if [ -z "${GPUS:-}" ]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
mkdir -p "$OUT/logs"
echo "verify: GPUs=[$GPUS] shards=$N out=$OUT sandbox=$SANDBOX tol=$TOL args=[$*]"

pids=()
for i in "${!GPU_ARR[@]}"; do
  g=${GPU_ARR[$i]}
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/verify_paper_weights.py --out "$OUT" --sandbox "$SANDBOX" --tol "$TOL" --shard "$i/$N" "$@" \
    > "$OUT/logs/shard-$i-gpu$g.log" 2>&1 &
  pids+=($!)
  echo "  shard $i/$N → GPU $g (pid ${pids[-1]}, log $OUT/logs/shard-$i-gpu$g.log)"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
"$PY" scripts/verify_paper_weights.py --out "$OUT" --summarize
echo "결과: $OUT/results.csv · $OUT/summary.md"
exit $rc
