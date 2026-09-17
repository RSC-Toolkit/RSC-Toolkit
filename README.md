# RSC-Toolkit

**Respiratory Sound Classification Toolkit** — re-implementations of respiratory sound classification methods
on **ICBHI 2017** and **SPRSound**, trained and evaluated through one shared data, feature and evaluation pipeline.

Published methods often differ in data split, metric and checkpoint-selection rule, which makes their numbers hard to
compare. RSC-Toolkit runs every method on the **official splits** with the same scoring code, so the contribution of a
method can be measured on equal terms.

- **34 registered models** — fine-tuning baselines (AST, BEATs, CNN6, ResNet, EfficientNet, CLAP), augmentation,
  contrastive/auxiliary objectives, foundation models and parameter-efficient fine-tuning.
- **Official protocols** — ICBHI 2017 official 60/40 split (4-class, respiratory cycles) and SPRSound 2022
  inter-patient test (7-class, events), plus ICBHI → SPRSound zero-shot evaluation.
- **Faithful ports** — golden-tensor regression tests pin each port to the authors' official code where it exists.
- **Released weights** — 340 trained runs (68 model families × 5 seeds) that can be downloaded and re-evaluated.
- **Extensible** — add your own model, loss, metric, feature processor or dataset without forking.

---

## Contents

1. [Installation](#installation)
2. [Quick start (CPU, a few minutes)](#quick-start-cpu-a-few-minutes)
3. [Datasets](#datasets)
4. [Training](#training)
5. [Evaluating a checkpoint](#evaluating-a-checkpoint)
6. [Released paper weights](#released-paper-weights)
7. [Reproducing a method](#reproducing-a-method)
8. [Adding your own model](#adding-your-own-model)
9. [Configuration reference](#configuration-reference)
10. [Testing](#testing)
11. [Known limitations](#known-limitations)
12. [License](#license)

---

## Installation

Python ≥ 3.9 (3.11 recommended).

```bash
git clone https://github.com/RSC-Toolkit/RSC-Toolkit.git
cd RSC-Toolkit
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[trainer,models]"
```

| extra | what it installs |
|---|---|
| `trainer` | Hydra, Lightning, TensorBoard — required for training |
| `models` | timm, transformers, OpenCV, resampy, torchlibrosa — model backbones and feature front ends |
| `notebook` | Jupyter, nbconvert, gdown — tutorial notebook and weight download |
| `ssl-weights` | lightning-bolts — only for the ResNet50 SimCLR weights |
| `dev` | pytest, black, flake8, isort |

To reproduce the exact environment used for the released weights:

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

For CPU-only use, install the CPU build of PyTorch first (about 1 GB instead of 6 GB):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
```

Check the installation:

```bash
rsctk --help          # dataset utilities and component listing
rsctk list models     # registered components by kind
```

The console scripts are `rsc-toolkit-train` (training), `rsc-toolkit-eval` (re-evaluation) and `rsctk` / `rsc-toolkit`
(datasets and listing).

## Quick start (CPU, a few minutes)

The repository ships a small synthetic toy dataset in the ICBHI file format (`sample_data/`). This trains a CNN6 for one epoch on the CPU and
re-evaluates the saved checkpoint:

```bash
rsctk datasets build-index toy --data_dir .datasets/toy --sample_level cycle

rsc-toolkit-train --config-name default data=toy_fbank model=cnn model.params.backbone=cnn6 callbacks=best_checkpoint \
  data.params.data_dir=.datasets/toy data.params.num_workers=0 \
  trainer.params.max_epochs=1 trainer.params.accelerator=cpu run.group=quickstart run.name=toy-cnn

rsc-toolkit-eval outputs/quickstart/toy-cnn --sandbox /tmp/rsc_eval_sandbox
```

[`TUTORIAL.md`](TUTORIAL.md) walks through the full path — toy data, registering your own model, one epoch on real
ICBHI data, and re-evaluating a released weight. The same content is available as a notebook,
[`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb), and can be run end to end with `scripts/tutorial/run_tutorial.sh`.

## Datasets

| dataset | task | split used |
|---|---|---|
| ICBHI 2017 | 4-class respiratory cycles (normal / crackle / wheeze / both) | official 60/40 train/test split |
| SPRSound (2022) | 7-class events | 2022 inter-patient test set |
| SPRSound (4-class view) | ICBHI label space, used for ICBHI → SPRSound zero-shot evaluation | 2022 inter-patient test set |

Download, index and validate:

```bash
rsctk datasets download    icbhi    --data_dir .datasets/icbhi
rsctk datasets build-index icbhi    --data_dir .datasets/icbhi --sample_level cycle
rsctk datasets validate    icbhi    --data_dir .datasets/icbhi

rsctk datasets download    sprsound --data_dir .datasets/sprsound
rsctk datasets build-index sprsound --data_dir .datasets/sprsound --sample_level cycle
rsctk datasets validate    sprsound --data_dir .datasets/sprsound
```

The configs expect the datasets under `.datasets/icbhi` and `.datasets/sprsound` relative to the working directory;
override `data.params.data_dir=...` to use another location. Raw recordings are never modified — splits, indices and
label mappings are written as derived files. In the SPRSound 4-class view, Rhonchi and Stridor events are excluded rather
than merged into wheeze (the 2022 inter-patient test set contains neither class).

Features are computed once and cached (`downloaded_datasets/features_cache` by default, or `RSCTK_FEATURE_CACHE_ROOT`).
The cache key includes every setting that affects the features, and each cache directory keeps a chain ledger
(`_chain.json`) so that a different preprocessing chain never silently reuses stale features.

## Training

Every experiment is one Hydra config plus optional overrides:

```bash
rsc-toolkit-train --config-name patchmix_icbhi seed=1 run.name=icbhi-patchmix-ce-s1
rsc-toolkit-train --config-name patchmix_icbhi model.params.method=patchmix_cl loss=cross_entropy \
  trainer.params.max_epochs=10 seed=3 run.name=my-run-s3
```

Outputs are written to `outputs/<group>/<run.name>/`:

- `best.pth` — weights of the selected checkpoint
- `best.json` — its metrics, e.g. `{"epoch", "S_p", "S_e", "icbhi_score"}` with `icbhi_score = (S_p + S_e) / 2`
- the Hydra snapshot of the resolved config (`outputs/<date>/<time>/.hydra/config.yaml`), used for re-evaluation

`scripts/train.sh <config> [overrides]` is a shell wrapper that selects the GPU (`GPU=<n>`) and caps per-worker
BLAS/OpenCV threads. A run refuses to start if `best.pth`/`best.json` already exist for the same `run.name`; choose a
new name or add `+run.overwrite=true`.

Smoke test (1 epoch, 2 batches):

```bash
scripts/train.sh patchmix_icbhi trainer.params.max_epochs=1 \
  ++trainer.params.limit_train_batches=2 ++trainer.params.limit_val_batches=2 run.name=smoke-patchmix
```

## Evaluating a checkpoint

```bash
rsc-toolkit-eval outputs/patchmix/icbhi-patchmix-ce-s1
rsc-toolkit-eval outputs/qlung/icbhi-qlung-s1 --override data.params.batch_size=16 --out eval.json
scripts/eval.sh outputs/bts/icbhi-bts-s1          # shell wrapper (GPU=<n>, sandbox)
```

`rsc-toolkit-eval` finds the run's config snapshot, rebuilds the model and data through the same factories used in
training, loads `best.pth` strictly and evaluates the test split. Training outputs are read-only (all writes go to the
sandbox, which must be outside `outputs/`). If `best.json` exists, the result is compared with it (PASS when |Δ| ≤ 0.05).

## Released paper weights

The weights behind the benchmark tables are released as run bundles (`best.pth`, `best.json`, `.hydra/config.yaml`,
`sha256.txt`). [`weights_manifest.csv`](weights_manifest.csv) lists every run with its metrics, checksum and download link.

```bash
pip install -e ".[notebook]"                                   # gdown
python scripts/download_paper_weights.py --list
python scripts/download_paper_weights.py --run icbhi-cnn6-ce-s1 --out paper_weights/
rsc-toolkit-eval paper_weights/icbhi-cnn6-ce-s1 --sandbox /tmp/rsc_eval_sandbox \
  --override data.params.data_dir=.datasets/icbhi
```

The downloader verifies every file against the bundle's `sha256.txt` and the manifest. Bundles carry their own config
snapshot, so no `outputs/` directory is needed.

Weights inherit the licence terms of the pre-trained backbone they were fine-tuned from (for example, OPERA-CT weights are
CC BY-NC). Weights built on backbones whose licence does not allow redistribution (the M2D family) are not released.

## Reproducing a method

Each method is one config; variants within a method (backbone, loss, view) are selected by overrides. Always add
`seed=N` and a matching `run.name=<name>-sN` (seeds 1–5 were used for the released weights).

```bash
# Example: Patch-Mix CL on ICBHI with seed 1, then re-evaluate the checkpoint
GPU=0 scripts/train.sh patchmix_icbhi model.params.method=patchmix_cl loss=cross_entropy seed=1 run.name=icbhi-patchmix-cl-s1
GPU=0 scripts/eval.sh  outputs/patchmix/icbhi-patchmix-cl-s1
```

| method (registered model) | ICBHI 2017 (official 60/40, 4-class) | SPRSound 2022 (inter-patient, 7-class) |
|---|---|---|
| AST fine-tuning (`ast_patchmix`) | `patchmix_icbhi model.params.method=ce` | `patchmix_sprs` (default `ce`) |
| Patch-Mix CE / CL (`ast_patchmix`) | `patchmix_icbhi model.params.method=patchmix` / `…=patchmix_cl` (+`loss=cross_entropy`) | `patchmix_sprs model.params.method=patchmix_cl` |
| SG-SCL / DAT (`ast_sgscl`) | `sgscl_icbhi` (default `sgscl`; benchmark rows use `model.params.con_feat=feat1`) / `model.params.method=dat` | — (SPRSound uses a single stethoscope, so the device-domain contrastive term is undefined) |
| ResNet18/34, EfficientNet-B0, CNN6 fine-tuning (`resnet18_sft`, …) | `sgscl_nonast_icbhi model.name=resnet18_sft` (`resnet34_sft`, `efficientnet_b0_sft`, `cnn6_sft`) | `sgscl_nonast_sprs model.name=…` |
| SG-SCL on non-AST backbones (`*_sft`) | `sgscl_nonast_icbhi model.name=cnn6_sft model.params.method=sgscl loss=cross_entropy` | — |
| SCL with metadata: CE / Hybrid / M-SCL (`cnn6_sclmeta`) | `sclmeta_icbhi model.params.method=ce` / `…=hybrid` / two stages: `sclmeta_icbhi_pretrain` → `sclmeta_icbhi_linear model.params.encoder_ckpt=<final_state.pth of the pretraining run>` | `sclmeta_sprs …` / `sclmeta_sprs_pretrain` → `sclmeta_sprs_linear` |
| Audio-CLAP fine-tuning (`audioclap`) | `bts_icbhi model=audioclap` | `bts_sprs model=audioclap` |
| BTS (`bts`) | `bts_icbhi` | `bts_sprs` |
| BTS-CARD (`bts_card`) | `btscard_icbhi trainer.params.max_epochs=30 optimizer.name=adamw` (paper setting) / `btscard_icbhi` (official-script setting) | `btscard_sprs` |
| BEATs fine-tuning / PAFA (`beats`) | `pafa_icbhi model.params.method=ce` / `pafa_icbhi` | `pafa_sprs model.params.method=ce` / `pafa_sprs` |
| PC-MCL (`beats_pcmcl`, `ast_pcmcl`) | `pcmcl_icbhi` / `pcmcl_ast_icbhi` | `pcmcl_sprs` (BEATs) / `pcmcl_ast_sprs` (AST) — toolkit extension with a 6-attribute label space |
| Lung-SRAD: FT / SAR / dual-axis (`dass_lungsrad`) | `lungsrad_icbhi model.params.method=ce` / `…=ce_blur` / `…=patchmix_cl model.params.cl_mode=dual model.params.mix_layout=as_paper` | `lungsrad_sprs …` |
| QLung, AST / Audio-CLAP (`ast_qlung`, `clap_qlung`) | `qlung_icbhi` / `qlung_clap_icbhi` | `qlung_sprs` / `qlung_clap_sprs` (toolkit extension) |
| AMS-D (`ast_amsd`) | `amsd_icbhi` (baseline: `model.params.method=baseline`) | `amsd_sprs` |
| AdD-RSC (`ast_addrsc`) | `addrsc_icbhi` (authors' split: `data.params.split_file=rsc_toolkit/datasets/icbhi/data/addrsc_random1_split.txt`) | `addrsc_sprs` |
| LungAdapter (`ast_lungadapter`) | `lungadapter_icbhi` (ablation, e.g. `model.params.use_res_adapter=false`) | `lungadapter_sprs` |
| LoRA-RSC (`ast_lora_rsc`) | `lorarsc_icbhi` | `lorarsc_sprs` |
| LungMix non-linear / none (`ast_lungmix`) | `lungmix_icbhi` / `model.params.variant=none` | `lungmix_sprs` |
| GaP-Aug with CNN14 (`cnn14_gapaug`) | `gapaug_icbhi` (naive baseline: `model.params.method=naive`) | `gapaug_sprs` (toolkit extension) |
| Domain Transfer, ResNeSt (`resnest_ibn_dt`) | `domaintransfer_icbhi` (baseline: `model.params.baseline=true model.params.mixup_alpha=0 data.params.splice_per_class=0 …`) | `domaintransfer_sprs` |
| Prototype, CNN8-Pt (`protonet`) | `protonet_icbhi` (CNN8 baseline: `model.params.method=cnn8`) | `protonet_sprs` |
| RespireNet CNN / +CBA (`respirenet`) | `respirenet_icbhi data.params.split_mode=official data.params.pad_seed=1 data.params.aug_seed=1` / add `data.params.aug_scale=1 data.params.cba_seed=1` (authors' fold 4: omit `split_mode`) | — |
| MVST, five views + fusion (`ast_mvst`) | `mvst_icbhi model.params.view=16` (for each of 16, 32, 64, 128, 256), then fuse with `scripts/reproduction/aggregate_mvst_results.py` | `mvst_sprs model.params.view=…` |
| M2D / M2D-X / M2D-Resp (`m2d`) | `m2d_icbhi` / `model.params.pretrained_path=pretrained_models/upstream/m2d/<ckpt>` (M2D-Resp also `model.params.input_size=[80,608]`) | `m2d_sprs …` |
| OPERA-CT frozen / fine-tuned (`opera_ct`) | `opera_embc_icbhi` / `opera_ft_icbhi` | `opera_embc_sprs` / `opera_ft_sprs` |
| DDE-MAE (`audiomae_dde_cls`) | `dde_icbhi_b2 data=icbhi_ddemae_16k` / `dde_icbhi_stage1 data=icbhi_ddemae_16k` → `dde_icbhi_stage2 data=icbhi_ddemae_16k model.params.encoder_ckpt=<stage-1 final_state.pth>` | — |

Data configs live in `rsc_toolkit/configs/data/` (`icbhi_*`, `sprsound_*`). "Toolkit extension" marks a setting the
original paper did not report; its recipe is the paper's ICBHI recipe applied unchanged.

**Pre-trained backbones.** Put checkpoints under `pretrained_models/upstream/` or point `RSCTK_PRETRAINED_ROOT` to a
directory that holds them. AST, PANNs and Hugging Face models (CLAP, AudioSet AST) are downloaded automatically; BEATs,
DASS, M2D, OPERA-CT and AudioMAE checkpoints must be downloaded manually — the toolkit stops with the expected path and
source when one is missing.

Per-method seed aggregation scripts are in `scripts/reproduction/aggregate_*.py`.

## Adding your own model

**No fork is needed.** Subclass `BaseRSCModel`, implement `training_forward`, and register the class; the training and
validation loops work unchanged.

```python
from torch import nn
from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import ModelFactory

@ModelFactory.register_model_class("my_model")
class MyModel(BaseRSCModel, nn.Module):
    def __init__(self, n_cls: int = 4, **params):
        super().__init__()
        self.net = ...          # n_cls is injected from model.params.n_cls

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.net(audio_batch)
        return logits, loss_func(logits, labels)   # return (logits, loss)
```

The full contract is documented in [`rsc_toolkit/models/base.py`](rsc_toolkit/models/base.py). Training and validation
call the same method; branch on `self.training` if they must differ. A model that consumes the third batch element
(metadata) declares `accepts_meta = ("dev", ...)`, and the adapter checks it against the datamodule's `meta_kind`
before training starts.

Register the module and provide a model config from outside the repository:

```bash
RSC_TOOLKIT_PLUGINS=my_pkg.my_model rsc-toolkit-train --config-name default --config-dir my_configs \
  data=icbhi_patchmix model=my_model
```

### Other extension points

Models, losses, metrics, optimizers, datamodules, raw datasets, adapters, feature processors, feature augmentations and
metadata labels all use the same registry mechanism ([`rsc_toolkit/registries.py`](rsc_toolkit/registries.py)).
`ModelFactory.register_model_class` is an alias of `MODELS.register`.

```python
from torch import nn
from rsc_toolkit.registries import LOSSES, METRICS, PROCESSORS, META_LABELS

@LOSSES.register("focal")                 # loss.name=focal
class FocalLoss(nn.Module): ...

@METRICS.register("hflung_jaccard")       # metrics: val: [{name: hflung_jaccard}]
class HFLungJaccard(Metric): ...

@PROCESSORS.register("hs_embed")          # data.params.features.kind=hs_embed
class HeartSoundSSLProcessor(BaseAudioProcessor):
    is_spectrogram = False                # True allows SpecAugment on training batches
    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):   # the features block (+ features.params)
        return cls(ckpt=opts["params"]["ckpt"], sampling_rate=sampling_rate, feature_cache=feature_cache)
    def get_processor_config(self):       # = feature cache key; include every setting that affects the output
        return {"kind": "hs_embed", "ckpt": self.ckpt, "sr": self.sampling_rate}

@META_LABELS.register("hospital")         # data.params.meta_label_mode=hospital (factory -> item -> int)
def build_hospital():
    return lambda item: int(item["meta"]["hospital"])
```

A custom index dataset subclasses `IndexDataset` and is injected through the config — no datamodule subclass is needed:

```yaml
data:
  params:
    datamodule_name: index
    index_dataset_cls: my_pkg.data.HeartSoundIndexDataset
    index_dataset_params: { bandpass_hz: [25, 400], fade: false }   # passed to the constructor as-is
    collate_fn: my_pkg.data.my_collate                               # optional dotted path
    features: { enabled: true, kind: hs_embed, params: { ckpt: /path/to/ssl.pt } }
```

There are three ways to register components:

1. **Inside the package** — modules in `rsc_toolkit/models/` are scanned automatically; other kinds are declared lazily
   in the registry hub.
2. **Entry point** — an installed package registers everything through one group:

   ```toml
   [project.entry-points."rsc_toolkit.plugins"]
   my_plugin = "my_pkg"          # a module whose import runs the decorators
   ```

   (The older group `rsc_toolkit.models` is still read.)
3. **Environment variable** — for local modules that are not packaged: `RSC_TOOLKIT_PLUGINS=my_pkg`.

Pass the registered name as `model.name`, `loss.name`, `features.kind`, and so on. Unknown names fail immediately with the
list of available ones. Your own configs are added with `--config-dir` (training) or
`rsctk datasets preprocess … --config_dir` (feature precomputation).

### Changing the training loop

Callbacks are plain Hydra `_target_` entries, so external classes can be attached directly (see
`rsc_toolkit/configs/callbacks/*.yaml`). Checkpoint selection, the best-metric criterion and resume behaviour are all
callbacks.

## Configuration reference

| environment variable | purpose |
|---|---|
| `RSCTK_PRETRAINED_ROOT` | directory with pre-trained checkpoints (looked up by the path without the `pretrained_models/` prefix, or by file name) |
| `RSCTK_FEATURE_CACHE_ROOT` | root of the feature cache |
| `RSCTK_PROJECT_ROOT` | repository root used to resolve relative paths |
| `RSC_TOOLKIT_PLUGINS` | comma-separated modules to import for registration |
| `RSCTK_NDE_HUB_FALLBACK=1` | allow BTS-CARD to fetch its text encoder from the Hugging Face hub when no local copy exists |

Relative paths in configs (`.datasets/…`, `outputs/…`, `pretrained_models/…`) are resolved against the working directory.

Repository layout:

```
rsc_toolkit/
  configs/        Hydra configs (one top-level config per method × dataset; data/, model/, callbacks/, …)
  datasets/       downloaders, index builders, datamodules, feature cache
  processors/     feature front ends (fbank, log-mel, CLAP, RespireNet images, …)
  models/         model implementations (auto-registered)
  methods/        training objectives shared by models
  augmentations/  waveform and spectrogram augmentations
  metrics/        ICBHI score (S_p, S_e, Score) and variants
  trainer/        Lightning adapters and callbacks
scripts/          train/eval wrappers, weight download, tutorial, reproduction utilities
tests/            unit and integration tests, golden references
sample_data/      synthetic toy dataset (ICBHI file format) used by the tutorial and CI
```

## Testing

```bash
pytest -m "not slow"      # fast regression tests (CPU)
pytest                    # everything, including golden-tensor tests (some need GPUs, datasets or checkpoints)
```

Tests that need unavailable datasets, checkpoints or CUDA kernels are skipped. CI (`.github/workflows/ci.yml`) runs the
wheel smoke test, the CPU unit tests and the tutorial on toy data.

## Known limitations

- Feature caches are keyed by preprocessing settings, not by dataset content; rebuild the cache if you replace the raw
  data in place.
- The wheel does not include `sample_data/`; with a plain `pip install` start from your own data or ICBHI.
- `scripts/packaging/smoke_installed_wheel.sh` checks an installed wheel from a clean virtual environment.

## License

RSC-Toolkit is released under the [PolyForm Noncommercial License 1.0.0](LICENSE): you may use, modify and share it for
research, education and other noncommercial purposes. Commercial use requires separate permission.

Code from the original RSC-Toolkit code base stays under the MIT License. Files that contain code ported from other
projects keep their original licenses (MIT, BSD-3-Clause, Apache-2.0,
CC BY-NC 4.0, CC BY-NC-SA 4.0); each such file names its source in a header comment, and the list with full license
texts is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES). Released weights inherit the
license of the pre-trained backbone they were fine-tuned from.
