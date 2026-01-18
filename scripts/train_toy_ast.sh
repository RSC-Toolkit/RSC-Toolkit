#!/usr/bin/env bash
set -euo pipefail

python -m rsc_toolkit.cli datasets build-index toy \
  --project_root . \
  --data_dir downloaded_datasets/toy \
  --sample_level cycle

python -m rsc_toolkit.train \
  data=toy_fbank \
  model=ast \
  data.params.data_dir=downloaded_datasets/toy \
  model.params.imagenet_pretrain=false \
  model.params.model_size=tiny224
