# 1. Setup

[Tutorial index](README.md) · next: [2. Toy data](02-toy-data.md)

Requirements: Python ≥ 3.9 (3.11 recommended).

## Install

```bash
git clone https://github.com/RSC-Toolkit/RSC-Toolkit.git RSC-Toolkit && cd RSC-Toolkit
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[trainer,models,notebook]"
```

- For the exact package versions used for the released weights, run `pip install -r requirements-lock.txt` and then
  `pip install -e . --no-deps` instead of the last line.
- CPU only (parts 2–3): install the CPU build of PyTorch **before** the last line to download about 1 GB instead of 6 GB:

  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
  ```

Check the installation:

```bash
python -m rsc_toolkit.cli --help
```

It prints the usage of `rsc-toolkit` with the sub-commands `datasets` and `list` (`rsctk --help` is the same).

## Working directory

The following parts write everything under one working directory. Training outputs are written to
`outputs/<group>/<run>/` under the current directory, so run the commands from `WORK`:

```bash
export WORK=$PWD/outputs/tutorial ROOT=$PWD
mkdir -p "$WORK" && cd "$WORK"
export RSCTK_PRETRAINED_ROOT=$WORK/pretrained                  # where downloaded backbone weights go
export SANDBOX=$(mktemp -d /tmp/rsc_tutorial_sandbox.XXXXXX)   # write isolation for re-evaluation (outside outputs/)
```

`outputs/` is ignored by git.

## Installing without a clone

If you only want to plug in your own model, the package can be installed directly:

```bash
pip install "rsc-toolkit[trainer,models] @ git+https://github.com/RSC-Toolkit/RSC-Toolkit.git"
```

All commands work the same way, but the toy data (`sample_data/`) is not included in the wheel, so start from your own
data or ICBHI ([part 4](04-icbhi.md)) instead of part 2.

---
next: [2. Toy data](02-toy-data.md)
