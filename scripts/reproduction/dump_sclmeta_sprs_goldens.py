"""
SCL-metadata 의 SPRSound 경로(dataset.py SPRS) P0 골든 — 공식 코드를 우리 venv 에서 실행.
  .venv/bin/python scripts/reproduction/dump_sclmeta_sprs_goldens.py --repo <sclmeta 클론> --data .datasets/sprsound/Classification --out tests/golden/sclmeta_sprs
골든: audio.pt (train/inter_test 이벤트 8개: 짧은/긴 것 포함 → 128000 샘플), rows.json (전 행: wav_path, onset, offset, split, event_label→id, sa_class_num).
"""
import argparse
import json
import pathlib
import sys
import types

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default=".datasets/sprsound/Classification")
    ap.add_argument("--out", default="tests/golden/sclmeta_sprs")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(args.repo).resolve())); sys.argv = [sys.argv[0]]
    sys.modules.setdefault("torchinfo", types.ModuleType("torchinfo")); sys.modules["torchinfo"].summary = lambda *a, **k: None
    from torchaudio import transforms as T
    from dataset import SPRS, SPRS_CLASS_DICT

    df = pd.read_csv(pathlib.Path(args.repo) / "data/SPRS/metadata.csv")
    rows = [{"wav_path": r.wav_path, "onset": int(r.onset), "offset": int(r.offset), "split": r.split,
             "label": SPRS_CLASS_DICT[r.event_label], "sa": int(r.sa_class_num), "gender": int(r.gender), "age": float(r.age)}
            for r in df.itertuples()]
    json.dump(rows, open(out / "rows.json", "w"))

    golden = []
    for split, subdir in (("train", "train_classification_wav"), ("inter_test", "valid_classification_wav/2022")):
        sub = df[df.split == split].copy(); sub["dur"] = sub["offset"] - sub["onset"]
        picks = pd.concat([sub.head(2), sub[sub["dur"] > 8000].head(1), sub.sort_values("dur").head(1)])
        ds = SPRS.__new__(SPRS)
        ds.data_path = str(pathlib.Path(args.data) / subdir); ds.samplerate, ds.duration = 16000, 8
        ds.targetsample = 128000; ds.pad_type, ds.meta_label = "circular", "sa"; ds.fade_samples = 1000
        ds.fade = T.Fade(fade_in_len=1000, fade_out_len=1000, fade_shape="linear"); ds.fade_out = T.Fade(fade_in_len=0, fade_out_len=1000, fade_shape="linear")
        ds.df = picks.reset_index(drop=True)
        data, labels, metas = ds.get_dataset()
        for i, r in enumerate(ds.df.itertuples()):
            golden.append({"wav_path": r.wav_path, "onset": int(r.onset), "offset": int(r.offset), "split": split,
                           "label": int(labels[i]), "meta": int(metas[i]), "wav": data[i, 0].clone()})
    torch.save(golden, out / "audio.pt")
    print("rows", len(rows), "| audio golden", len(golden), [(g["wav_path"], g["offset"] - g["onset"]) for g in golden[:3]])


if __name__ == "__main__":
    main()
