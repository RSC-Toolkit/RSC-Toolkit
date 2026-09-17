#!/usr/bin/env bash
# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
# Trains a small AST (tiny224, no pretraining) on the bundled toy data for 1 epoch on CPU — installation check. Full guide: tutorial/README.md
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=$(command -v python3)   # works even if the venv is not on PATH
"$PY" -m rsc_toolkit.cli datasets build-index toy --project_root "$ROOT" --data_dir downloaded_datasets/toy --sample_level cycle
"$PY" -m rsc_toolkit.train --config-name default data=toy_fbank model=ast callbacks=best_checkpoint \
  model.params.imagenet_pretrain=false model.params.model_size=tiny224 \
  trainer.params.max_epochs="${EPOCHS:-1}" trainer.params.accelerator="${ACCEL:-cpu}" data.params.num_workers=0 \
  run.name=toy-ast run.group=toy "$@"
