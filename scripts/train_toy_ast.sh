#!/usr/bin/env bash
# 번들 toy 데이터로 소형 AST(tiny224, 사전학습 없음)를 CPU 에서 1 epoch 학습 — 설치 확인용. 전체 안내: docs/TUTORIAL.md
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=$(command -v python3)   # venv 가 PATH 에 없어도 동작
"$PY" -m rsc_toolkit.cli datasets build-index toy --project_root "$ROOT" --data_dir downloaded_datasets/toy --sample_level cycle
"$PY" -m rsc_toolkit.train --config-name default data=toy_fbank model=ast callbacks=best_checkpoint \
  model.params.imagenet_pretrain=false model.params.model_size=tiny224 \
  trainer.params.max_epochs="${EPOCHS:-1}" trainer.params.accelerator="${ACCEL:-cpu}" data.params.num_workers=0 \
  run.name=toy-ast run.group=toy "$@"
