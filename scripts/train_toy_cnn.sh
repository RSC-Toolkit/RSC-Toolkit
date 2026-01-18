#!/usr/bin/env bash
set -euo pipefail

python -m rsc_toolkit.cli datasets build-index toy \
  --project_root . \
  --data_dir downloaded_datasets/toy \
  --sample_level cycle

python -m rsc_toolkit.train \
  data=toy_fbank \
  model=cnn \
  data.params.data_dir=downloaded_datasets/toy
