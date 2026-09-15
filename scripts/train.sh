#!/bin/bash
# 선택한 config(=논문 레시피)로 학습 한 건을 돌린다 — rsc-toolkit-train 의 얇은 셸 래퍼.
#
#   scripts/train.sh <config-name> [hydra overrides...]
#
#   예) scripts/train.sh patchmix_icbhi seed=1 run.name=my-patchmix-s1
#       GPU=3 scripts/train.sh sgscl_nonast_icbhi model.name=resnet18_sft data=icbhi_sgscl_nonast seed=2
#       scripts/train.sh m2d_icbhi trainer.params.max_epochs=1 ++trainer.params.limit_train_batches=2   # 스모크
#
# 모델·데이터는 config 그룹 override 로 고른다: model=<configs/model/*.yaml 이름> data=<configs/data/*.yaml 이름>
# (또는 model.name=<등록 이름>). 등록된 이름 목록: rsc-toolkit list (없으면 python -c "import rsc_toolkit.registries as r; print(sorted(r.MODELS))").
# 결과: outputs/<계열>/<run.name>/best.pth·best.json (계열은 config 의 callbacks dirpath), hydra 스냅샷 outputs/<날짜>/<시각>/.hydra.
#
# 환경변수: GPU(기본: 환경의 CUDA_VISIBLE_DEVICES, 없으면 0) · PY(기본 .venv/bin/python) · THREAD_CAP(기본 1 — 동시 다중 run 시 dataloader
#           워커의 OpenCV/OpenBLAS 스레드 폭주 방지; 0 이면 캡 안 함)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
GPU=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}   # GPU 미지정이면 환경의 CUDA_VISIBLE_DEVICES 를 존중 (예전엔 항상 0 으로 덮었다)
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
