# 4. ICBHI 2017 (GPU)

[Tutorial index](README.md) · previous: [3. Your own model](03-your-own-model.md) · next: [5. Released weights](05-released-weights.md)

Uses `WORK`, `ROOT`, `SANDBOX` and `RSCTK_PRETRAINED_ROOT` from [part 1](01-setup.md#working-directory); run the commands
from `WORK`. Needs one GPU and about 8 GB of disk (ICBHI zip 1.98 GB, extracted 3.9 GB, feature cache 2.7 GB); about
30 minutes depending on download speed.

## Download, index, validate

```bash
rsctk datasets download    icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi"
rsctk datasets build-index icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi" --sample_level cycle
rsctk datasets validate    icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi"
```

The index uses the official 60/40 split: 920 recordings (539 train / 381 test), 6,898 respiratory cycles.

## Train one epoch of a paper recipe

The CNN6 fine-tuning recipe (`sgscl_nonast_icbhi` with `model.name=cnn6_sft`) normally runs 400 epochs; one epoch is enough
to check the path.

For GPU training, cap BLAS/OpenCV threads per dataloader worker at 1 (this avoids worker × thread-pool oversubscription on
shared machines):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
```

The PANNs CNN6 pre-trained weights are downloaded automatically from Zenodo (record 3960586) to
`$RSCTK_PRETRAINED_ROOT/upstream/Cnn6_mAP=0.343.pth`. If Zenodo is blocked on your network, place the file there yourself.

```bash
rsc-toolkit-train --config-name sgscl_nonast_icbhi model.name=cnn6_sft \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/icbhi" data.params.features.cache_dir="$WORK/cache_icbhi" \
  trainer.params.max_epochs=1 run.name=icbhi-cnn6-1epoch run.group=tutorial hydra.run.dir="$WORK/hydra/icbhi-cnn6-1epoch" +run.overwrite=true
```

This recipe's callbacks save to `$WORK/outputs/sgscl/<run.name>`:

```bash
rsc-toolkit-eval "$WORK/outputs/sgscl/icbhi-cnn6-1epoch" --project-root "$ROOT" --hydra-dir "$WORK/hydra/icbhi-cnn6-1epoch" --sandbox "$SANDBOX"
```

Other methods follow the same pattern with the configs in the [README recipe table](../README.md#reproducing-a-method).

---
previous: [3. Your own model](03-your-own-model.md) · next: [5. Released weights](05-released-weights.md)
