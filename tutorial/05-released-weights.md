# 5. Released weights (GPU)

[Tutorial index](README.md) · previous: [4. ICBHI 2017](04-icbhi.md)

Uses `WORK`, `ROOT` and `SANDBOX` from [part 1](01-setup.md#working-directory) and the ICBHI data from [part 4](04-icbhi.md).

The weights behind the benchmark tables are released as run bundles (`best.pth`, `best.json`, `.hydra/config.yaml`,
`sha256.txt`). [`weights_manifest.csv`](../weights_manifest.csv) lists all 340 runs with metrics, checksums and Google Drive
links. `scripts/download_paper_weights.py` downloads a bundle with gdown (included in the `notebook` extra) and verifies its
sha256 checksums.

## What is released and how it is organised

- 340 runs = 68 model families × seeds 1–5. Run names are `<family>-s<seed>` (families start with `icbhi-` or `sprs-`);
  seed 1 of `icbhi-ast-sft`, `icbhi-patchmix-ce` and `icbhi-patchmix-cl` has no `-s1` suffix.
- One shared Google Drive folder holds all bundles in the layout
  `weights_release/<icbhi|sprsound>/<family>/<run>/`, each with `best.pth`, `best.json`, `.hydra/config.yaml`, `README.txt`
  and `sha256.txt` (MVST fusion runs have no `.hydra/`).
- `best.pth` sizes range from 12 MB (MVST fusion heads) and 17 MB (CNN6) to 670 MB (BTS-CARD) per run; all 340 runs
  together are 87 GB.

List every run with its dataset, Score, size and link:

```bash
python "$ROOT/scripts/download_paper_weights.py" --list
```

## Download

`--run` selects one run and can be repeated. `--out` is the target directory; each run goes to `<out>/<run>/`, and all
files are checked against the bundle's `sha256.txt` and the manifest.

```bash
python "$ROOT/scripts/download_paper_weights.py" --run icbhi-cnn6-ce-s1 --out "$WORK/paper_weights"
```

What to expect with the current downloader:

- For every requested run, the script first reads the listing of the whole shared folder (about 740 subfolders). This takes
  about 6 minutes per run; the files themselves then download quickly (350 MB in about 30 s).
- After many requests Google Drive may refuse for a while (`status code 500 … have had many accesses`, or
  `… but Gdown can't. Please check connections and permissions`). Wait and run the same command again, or download in the
  browser as in example 2.
- A run that is already present in `<out>/<run>/` is not downloaded again; the script only checks its checksums and
  prints `already present, all sha256 match`.
- For more than a few runs, the browser is faster: you download whole family folders at once and use the script only for
  verification.

### Example 1: a few runs with the script

Seeds 1 and 2 of the CNN6 fine-tuning family (each run reads the folder listing once, so allow several minutes per run):

```bash
python "$ROOT/scripts/download_paper_weights.py" --run icbhi-cnn6-ce-s1 --run icbhi-cnn6-ce-s2 --out "$WORK/paper_weights"
```

### Example 2: a whole family in the browser, verified with the script

1. Open the Google Drive link shown by `--list` (the same for every run) and go to
   `weights_release/icbhi/icbhi-patchmix-ce/`.
2. Download the folder `icbhi-patchmix-ce` (Drive packs it into a zip file) and unzip it.
3. Move the five run folders so that each one sits directly under `$WORK/paper_weights/`, for example
   `$WORK/paper_weights/icbhi-patchmix-ce-s2/best.pth`. Keep the hidden `.hydra/` folders.
4. Verify all five; nothing is downloaded when the checksums match:

```bash
python "$ROOT/scripts/download_paper_weights.py" --out "$WORK/paper_weights" \
  --run icbhi-patchmix-ce --run icbhi-patchmix-ce-s2 --run icbhi-patchmix-ce-s3 --run icbhi-patchmix-ce-s4 --run icbhi-patchmix-ce-s5
```

If a check fails (for example `.hydra/config.yaml missing`), the script tries to download that run again; fix the folder
layout first if you want to avoid that.

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
