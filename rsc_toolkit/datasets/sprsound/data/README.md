# SPRSound bundled data

This directory holds the SPRSound auxiliary files shipped with the package.
The original audio is not stored here -- download it with `rsctk datasets download sprsound --data_dir <path>`
from the official GitHub repository (SJTU-YONGFU-RESEARCH-GRP/SPRSound).

## Files

- `metadata.csv` -- event-level annotation table used to build the index.
  Generated from the original json annotations with `generate_annotation.py`:
  - train: `Classification/train_classification_json/*.json`
  - valid: `Classification/valid_classification_json/2022/inter_test_json/*.json` (2022 inter-test only)
  - `start`/`end` converted from ms to seconds, `Wheeze+Crackle` label merged into `Both`
  - columns: Filename, Patient_number, Age, Gender, Recording_location, Recording_number,
    Record_annotation, Start, End, Label, Split
- `generate_annotation.py` -- the script that produced the `metadata.csv` above (kept for provenance).
  It assumes `Classification/` is in the current directory, so adjust paths when regenerating.

## Notes

- The upstream repository is downloaded as the main-branch zip, so its content may grow over time.
  The bundled `metadata.csv` covers only train + the 2022 inter-test.
- A gdown-based script that downloads from a Google Drive mirror is not included:
  the official downloader does the same, and gdown is not a project dependency.
- The mirror in question (source and snapshot date unverified):
  https://drive.google.com/file/d/1rgq1ev7yk-vPTnlfJW3K_eWj-cy3ywL8/view?usp=sharing
  Kept only as a manual fallback candidate when the official repository is unreachable. The data snapshot is unverified,
  so for reproduction prefer the official repository (`rsctk datasets download sprsound`).
