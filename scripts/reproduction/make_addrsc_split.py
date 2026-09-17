"""
Reconstructs the ADD-RSC code split — util/icbhi_dataset.py:35-45:
  filenames = sorted({stem of *.wav/*.txt in data_folder}); indices = range(920); random.Random(1).shuffle(indices);
  train = first int(920*0.6)=552 files, test = remaining 368 (file-level, independent of patients and the official split).
Assumes the data folder contains only recordings (920 wav+txt) (with any other txt the original code fails when loading the wav). The same set is built from the recording_path stems of the toolkit index.

  .venv/bin/python scripts/reproduction/make_addrsc_split.py [--index .datasets/icbhi/index/index_cycle.csv] [--out rsc_toolkit/datasets/icbhi/data/addrsc_random1_split.txt]
"""
import argparse
import pathlib
import random

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=".datasets/icbhi/index/index_cycle.csv")
    ap.add_argument("--out", default="rsc_toolkit/datasets/icbhi/data/addrsc_random1_split.txt")
    args = ap.parse_args()
    df = pd.read_csv(args.index)
    filenames = sorted({pathlib.Path(p).stem for p in df["recording_path"]})
    assert len(filenames) == 920, len(filenames)
    indices = list(range(len(filenames)))
    random.Random(1).shuffle(indices)
    train_size = int(len(indices) * 0.6)
    train = {filenames[i] for i in indices[:train_size]}
    out = pathlib.Path(args.out)
    lines = ["# ADD-RSC util/icbhi_dataset.py code split: sorted stems -> random.Random(1).shuffle -> first 60% train (file level). scripts/reproduction/make_addrsc_split.py"]
    lines += [f"{f}\t{'train' if f in train else 'test'}" for f in filenames]
    out.write_text("\n".join(lines) + "\n")
    n_cycles = df["recording_path"].map(lambda p: pathlib.Path(p).stem in train).sum()
    print(f"train files {len(train)} / test {len(filenames) - len(train)}; train cycles {n_cycles} / test {len(df) - n_cycles} → {out}")


if __name__ == "__main__":
    main()
