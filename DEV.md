# RSC-Toolkit (Respiratory Sound Classification Toolkit)

## Purpose
This project is a Python toolkit for respiratory sound classification focused on
data handling. The near-term scope is dataset downloading, processing, and
dataloading. Training and model execution are out of scope for now.

Goals:
- Replicate lab experiments reliably with readable configs.
- Provide reusable tools for the community to build and evaluate models.
- Keep the code clean, testable, and easy to extend.

## Current focus
- Dataset download, validation, and preprocessing.
- Feature extraction and caching (including model-dependent processors).
- PyTorch Dataset/DataLoader integration.
- Tests that cover download -> process -> load.

## Key components
- `rsc_toolkit/cli.py`: CLI for dataset utilities (download/preprocess).
- `rsc_toolkit/datasets/`: Dataset abstractions, ICBHI implementation, and
  PyTorch Dataset/DataLoader wrappers.
- `rsc_toolkit/processors/`: Audio feature processors (e.g., mel/filterbank,
  CLAP) and caching hooks.
- `sample_data/toy_dataset/`: Minimal ICBHI subset used in tests.
- `tests/`: Dataset/processor integration tests and unit tests.

## Typical workflows
- Dataset setup: `rsc-toolkit datasets download icbhi --datasets_download_path ./datasets` (per README).
- Preprocess (feature caching): `rsc-toolkit datasets preprocess icbhi --datasets_download_path ./datasets`.
- Development install: `pip install -e .[dev]`.
- Tests: `pytest` (some dataset tests require `sample_data/toy_dataset`).

## Design requirements (data pipeline)
- Large datasets: support on-the-fly processing and optional full precompute.
- Sample levels: support "cycle" and "clip" access without duplicating logic.
- Caching: cache per dataset + processor config + model/encoder settings + sample level.
- Multiple datasets: shared base interface with dataset-specific download/process.
- Independence: dataset pipeline should not depend on trainer/model code.
- Config: use readable, reproducible config files (Hydra optional, not required).
- Observability: logging, minimal metrics, and traceable artifacts.

## Notes for AI dev agents
- The README and code disagree slightly on CLI script names; console scripts are
  defined in `pyproject.toml` as `rsc-toolkit-datasets` and `rsc-toolkit-train`.
- Dataset processing uses a cache under the outputs directory
  (see `rsc_toolkit/datasets/base.py`).
- The ICBHI dataset downloader expects to fetch the official dataset archive and
  metadata URLs; the toy dataset is bundled locally for tests.
