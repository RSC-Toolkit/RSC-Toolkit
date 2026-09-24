"""
Reproduces the recording split of MVST (wentaoheunnc/MVST util/icbhi_dataset.py:35-45):
  stems = sorted({f.split('.')[0] for f in listdir(data_folder) if '.wav' in f or '.txt' in f})
  indices shuffled with random.Random(1) → first 60% train, the rest test.
The result depends on whether auxiliary txt files were present in the authors' folder, so the stem set can be adjusted with --extra-stems.

  .venv/bin/python scripts/reproduction/make_mvst_split.py --data .datasets/icbhi --out <split.txt> [--extra-stems a b]
Output: toolkit split file format (one `<recording_id> train|test` per line). Stems that are not recordings are excluded from the output (they do not exist).
"""
import argparse
import os
import random


def mvst_split(stems, seed=1, train_ratio=0.6):
    filenames = sorted(set(stems))
    indices = list(range(len(filenames)))
    random.Random(seed).shuffle(indices)
    train_size = int(len(indices) * train_ratio)
    train = {filenames[i] for i in indices[:train_size]}
    return {f: ("train" if f in train else "test") for f in filenames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".datasets/icbhi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recordings-only", action="store_true", help="only recording stems that have a wav (exclude auxiliary txt)")
    ap.add_argument("--extra-stems", nargs="*", default=[], help="add stems of auxiliary files that were presumably in the authors' folder")
    args = ap.parse_args()
    files = os.listdir(args.data)
    stems = {f.strip().split(".")[0] for f in files if ".wav" in f or ".txt" in f}
    recs = {f[:-4] for f in files if f.endswith(".wav")}
    if args.recordings_only:
        stems = set(recs)
    stems |= set(args.extra_stems)
    split = mvst_split(stems)
    with open(args.out, "w") as fh:
        for f in sorted(split):
            if f in recs:
                fh.write(f"{f} {split[f]}\n")
    n_tr = sum(1 for f in recs if split.get(f) == "train")
    print(f"stems {len(stems)} (recordings {len(recs)}) → train {n_tr} / test {len(recs) - n_tr} recordings → {args.out}")


if __name__ == "__main__":
    main()
