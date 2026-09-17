# 5. Released weights (GPU)

[Tutorial index](README.md) · previous: [4. ICBHI 2017](04-icbhi.md)

Uses `WORK`, `ROOT` and `SANDBOX` from [part 1](01-setup.md#working-directory) and the ICBHI data from [part 4](04-icbhi.md).

The weights behind the benchmark tables are released as run bundles (`best.pth`, `best.json`, `.hydra/config.yaml`,
`sha256.txt`). [`weights_manifest.csv`](../weights_manifest.csv) lists all 340 runs with metrics, checksums and Google Drive
links. `scripts/download_paper_weights.py` downloads a bundle with gdown (included in the `notebook` extra) and verifies its
sha256 checksums.

## List and download

```bash
python "$ROOT/scripts/download_paper_weights.py" --list
```

```bash
python "$ROOT/scripts/download_paper_weights.py" --run icbhi-cnn6-ce-s1 --out "$WORK/paper_weights"
```

All bundles are in one shared Drive folder, so listing it can take a few minutes; the downloader retries when Drive
answers with a temporary error.

## Re-evaluate

```bash
rsc-toolkit-eval "$WORK/paper_weights/icbhi-cnn6-ce-s1" --project-root "$ROOT" --sandbox "$SANDBOX" \
  --override data.params.data_dir="$WORK/icbhi" --override data.params.features.cache_dir="$WORK/cache_icbhi"
```

The bundle carries its own config snapshot, so no `outputs/` directory is needed. Because the data paths are overridden,
the result is reported as `status: OK` together with |Δ| against `best.json` (the released value); on the same official
test set |Δ| is 0.

The same bundle can also be evaluated zero-shot on SPRSound (after downloading and indexing SPRSound as in the
[README](../README.md#datasets)):

```bash
python "$ROOT/scripts/eval_zero_shot_sprsound.py" "$WORK/paper_weights/icbhi-cnn6-ce-s1" --project-root "$ROOT" \
  --sprsound-dir "$ROOT/.datasets/sprsound" --sandbox "$SANDBOX"
```

---
previous: [4. ICBHI 2017](04-icbhi.md) · [Tutorial index](README.md)
