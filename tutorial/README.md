# RSC-Toolkit tutorial

A minimal, step-by-step path through the toolkit. Each part is short and builds on the previous one.

| part | what you do | needs | time |
|---|---|---|---|
| [1. Setup](01-setup.md) | install the toolkit and prepare a working directory | Python 3.11 | ~5 min |
| [2. Toy data](02-toy-data.md) | build an index, train a CNN6 for one epoch, re-evaluate it | CPU | a few minutes |
| [3. Your own model](03-your-own-model.md) | register a model from outside the repository and train it | CPU | a few minutes |
| [4. ICBHI 2017](04-icbhi.md) | download the official data and train one epoch of a paper recipe | 1 GPU, ~8 GB disk | ~30 min |
| [5. Released weights](05-released-weights.md) | download a released checkpoint and re-evaluate it | 1 GPU, part 4 data | ~10 min |

Parts 2–5 use the shell variables defined in part 1 (`WORK`, `ROOT`, `SANDBOX`, `RSCTK_PRETRAINED_ROOT`); run them in the
same shell, or define the variables again.

The same content is available as a notebook, [`notebooks/tutorial.ipynb`](../notebooks/tutorial.ipynb), and can be run end
to end with `scripts/tutorial/run_tutorial.sh` (parts 2–3; add `--full` for parts 4–5). CI runs it on the toy data.

## Tools at a glance

| task | command |
|---|---|
| train a method recipe | `scripts/train.sh <config> [override…]` (see the recipe table in the [README](../README.md#reproducing-a-method)) |
| re-evaluate one run | `rsc-toolkit-eval outputs/<group>/<run>` |
| zero-shot evaluation on SPRSound | `python scripts/eval_zero_shot_sprsound.py <run dir>` |
| run this tutorial automatically | `scripts/tutorial/run_tutorial.sh [--full]` |
