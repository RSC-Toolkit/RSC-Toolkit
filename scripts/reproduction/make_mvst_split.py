"""
MVST(wentaoheunnc/MVST util/icbhi_dataset.py:35-45) 의 녹음 스플릿 재현:
  stems = sorted({f.split('.')[0] for f in listdir(data_folder) if '.wav' in f or '.txt' in f})
  indices 를 random.Random(1).shuffle → 앞 60% train, 나머지 test.
저자 폴더의 부속 txt 유무에 따라 결과가 달라지므로 --extra-stems 로 stem 집합을 조정할 수 있다.

  .venv/bin/python scripts/reproduction/make_mvst_split.py --data .datasets/icbhi --out <split.txt> [--extra-stems a b]
출력: 툴킷 split 파일 형식 (`<recording_id> train|test` 한 줄씩). 녹음이 아닌 stem 은 출력에서 제외(존재하지 않음).
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
    ap.add_argument("--recordings-only", action="store_true", help="wav 가 있는 녹음 stem 만 (부속 txt 제외)")
    ap.add_argument("--extra-stems", nargs="*", default=[], help="저자 폴더에 있었을 부속 파일 stem 을 추가")
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
    print(f"stems {len(stems)} (녹음 {len(recs)}) → train {n_tr} / test {len(recs) - n_tr} 녹음 → {args.out}")


if __name__ == "__main__":
    main()
