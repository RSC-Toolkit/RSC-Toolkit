#!/usr/bin/env bash
# 번들 toy 데이터(ICBHI 환자 107)로 CNN 을 CPU 에서 1 epoch 학습 — 설치 확인용. 전체 안내: docs/TUTORIAL.md · notebooks/tutorial.ipynb
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=$(command -v python3)   # venv 가 PATH 에 없어도 동작
"$PY" -m rsc_toolkit.cli datasets build-index toy --project_root "$ROOT" --data_dir downloaded_datasets/toy --sample_level cycle
"$PY" -m rsc_toolkit.train --config-name default data=toy_fbank model=cnn callbacks=best_checkpoint \
  trainer.params.max_epochs="${EPOCHS:-1}" trainer.params.accelerator="${ACCEL:-cpu}" data.params.num_workers=0 \
  run.name=toy-cnn run.group=toy "$@"
