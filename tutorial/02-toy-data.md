# 2. Toy data (CPU)

[Tutorial index](README.md) · previous: [1. Setup](01-setup.md) · next: [3. Your own model](03-your-own-model.md)

Uses the variables `WORK`, `ROOT` and `SANDBOX` from [part 1](01-setup.md#working-directory); run the commands from `WORK`.

The bundled toy data (`sample_data/toy_dataset/`) has the ICBHI file layout — recordings, cycle annotations and a split
file — but synthetic audio, so the whole pipeline runs without downloading anything. Scores on it are meaningless.

## Build the index

```bash
rsctk datasets build-index toy --project_root "$ROOT" --data_dir "$WORK/toy" --sample_level cycle
```

The toolkit never modifies the audio; it writes a derived index (`$WORK/toy/index/index_cycle.csv`).

## Train a CNN6 for one epoch

```bash
rsc-toolkit-train --config-name default data=toy_fbank model=cnn model.params.backbone=cnn6 callbacks=best_checkpoint \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/toy" data.params.features.cache_dir="$WORK/cache_toy" \
  data.params.num_workers=0 trainer.params.max_epochs=1 trainer.params.accelerator=cpu \
  run.name=toy-cnn run.group=tutorial hydra.run.dir="$WORK/hydra/toy-cnn" +run.overwrite=true
```

```bash
cat "$WORK/outputs/tutorial/toy-cnn/best.json"
```

- `callbacks=best_checkpoint` writes `best.pth` and `best.json` for the epoch with the best validation Score to
  `outputs/<run.group>/<run.name>/`.
- `hydra.run.dir` is where the config snapshot (`.hydra/config.yaml`) is written; re-evaluation uses it.
- `+run.overwrite=true` allows re-running: without it, the toolkit refuses to overwrite an existing `best.pth` of the
  same `run.name`.

## Re-evaluate the checkpoint

```bash
rsc-toolkit-eval "$WORK/outputs/tutorial/toy-cnn" --project-root "$ROOT" --hydra-dir "$WORK/hydra/toy-cnn" --sandbox "$SANDBOX"
```

The snapshot is used to rebuild the model and data through the same factories as in training; `best.pth` is loaded
strictly, evaluated, and compared with `best.json` (`PASS` when |Δ| ≤ 0.05). Pass the run directory as an absolute path
(relative paths are resolved against `--project-root`). All writes go to `--sandbox`, which must be outside `outputs/`.

---
previous: [1. Setup](01-setup.md) · next: [3. Your own model](03-your-own-model.md)
