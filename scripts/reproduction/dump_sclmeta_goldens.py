"""
SCL-metadata (ilyassmoummad/scl_icbhi2017) official-code parity goldens — runs the official code in our venv (only torch/torchaudio needed).

  .venv/bin/python scripts/reproduction/dump_sclmeta_goldens.py --repo <clone> --data .datasets/icbhi --out tests/golden/sclmeta

Goldens:
- audio.pt        : a few train cycles from metadata.csv (including short ones and ones over 8 s) through the dataset.py path (segment load→resample→fade→circular→fade_out) → 128000 samples
- features.pt     : batch of the waveforms above → per-sample dB-mel (cache stage), val_transform output (batch min-max + Standardize), two views of the fixed-seed train_transform
- cnn6.pt         : state_dict shapes of the CNN6 with partially loaded PANN weights + val features → 512-d embeddings
- losses.pt       : SupCon (labels/None), SupConCE (α .5, weights) and the M-SCL combined scalar on fixed proj/labels/meta
- meta_sa.json    : (filepath, onset, offset) → sa_class_num for every row of metadata.csv (for comparison with the toolkit meta 'sa')
"""
import argparse
import json
import pathlib
import sys
import types

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default=".datasets/icbhi")
    ap.add_argument("--out", default="tests/golden/sclmeta")
    ap.add_argument("--panns", default="pretrained_models/upstream/Cnn6_mAP=0.343.pth")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    # args.py parses argparse at import time — clear argv so it parses with the defaults
    sys.argv = [sys.argv[0]]
    sys.modules.setdefault("torchinfo", types.ModuleType("torchinfo"))
    sys.modules["torchinfo"].summary = lambda *a, **k: types.SimpleNamespace(trainable_params=0)

    import pandas as pd
    from torchaudio import transforms as T
    from dataset import ICBHI  # noqa: E402
    from utils import Normalize, Standardize  # noqa: E402
    from augmentations import SpecAugment  # noqa: E402
    from models import CNN6, Projector  # noqa: E402
    from losses import SupConLoss, SupConCELoss  # noqa: E402

    # ---- 1) audio: run dataset.py get_sample + the padding path on a few rows ----
    df = pd.read_csv(repo / "data/ICBHI/metadata.csv")
    df["dur"] = df["offset"] - df["onset"]
    tr = df[df["split"] == "train"]
    picks = pd.concat([tr.head(3), tr[tr["dur"] > 8].head(2), tr[tr["dur"] < 1].head(1)])
    ds = ICBHI.__new__(ICBHI)
    ds.data_path, ds.samplerate, ds.duration = args.data, 16000, 8
    ds.targetsample = 16000 * 8
    ds.pad_type, ds.meta_label = "circular", "sa"
    ds.fade_samples = 1000
    ds.fade = T.Fade(fade_in_len=1000, fade_out_len=1000, fade_shape="linear")
    ds.fade_out = T.Fade(fade_in_len=0, fade_out_len=1000, fade_shape="linear")
    ds.df = picks.reset_index(drop=True)
    data, labels, metas = ds.get_dataset()
    audio_golden = [{"filepath": r.filepath, "onset": float(r.onset), "offset": float(r.offset), "label": int(l),
                     "meta": int(m), "wav": data[i, 0].clone()} for i, (r, l, m) in enumerate(zip(ds.df.itertuples(), labels, metas))]
    torch.save(audio_golden, out / "audio.pt")
    print("audio golden", len(audio_golden), [a["filepath"] for a in audio_golden[:2]], tuple(data.shape))

    # ---- 2) features ----
    melspec = T.MelSpectrogram(n_fft=1024, n_mels=64, win_length=1024, hop_length=512, f_min=50, f_max=2000)
    normalize = Normalize()
    standardize = Standardize()
    specaug = SpecAugment(freq_mask=20, time_mask=40, freq_stripes=2, time_stripes=2)
    train_transform = torch.nn.Sequential(torch.nn.Sequential(melspec, normalize), specaug, standardize)
    val_transform = torch.nn.Sequential(torch.nn.Sequential(melspec, normalize), standardize)
    x = data  # (N, 1, 128000)
    with torch.no_grad():
        dbmel = normalize.power_to_db(melspec(x))          # per-sample cache stage
        val = val_transform(x)
        torch.manual_seed(5)
        v1 = train_transform(x)
        v2 = train_transform(x)
    torch.save({"wav": x, "dbmel": dbmel, "val": val, "seed": 5, "view1": v1, "view2": v2}, out / "features.pt")
    print("features", tuple(dbmel.shape), tuple(val.shape), "views equal:", torch.equal(v1, v2))

    # ---- 3) CNN6 ----
    enc = CNN6(num_classes=4, do_dropout=False, embed_only=True, from_scratch=False, path_to_weights=args.panns, device="cpu").eval()
    with torch.no_grad():
        emb = enc(val)
    torch.save({"state_dict_shapes": {k: list(v.shape) for k, v in enc.state_dict().items()}, "emb": emb}, out / "cnn6.pt")
    print("cnn6 emb", tuple(emb.shape), "keys", len(enc.state_dict()))

    # ---- 4) losses ----
    torch.manual_seed(11)
    p1, p2 = torch.randn(8, 128), torch.randn(8, 128)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]); meta = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pred1, pred2 = torch.randn(8, 4), torch.randn(8, 4)
    w = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32); w = 1.0 / (w / w.sum()); w = w / w.sum()
    sup = SupConLoss(temperature=0.06, device="cpu")
    supce = SupConCELoss(weights=w, alpha=0.5, device="cpu", temperature=0.06)
    losses = {"p1": p1, "p2": p2, "y": y, "meta": meta, "pred1": pred1, "pred2": pred2, "weights": w,
              "supcon_labels": sup(p1, p2, y).detach(), "supcon_none": sup(p1, p2, None).detach(),
              "supconce": supce(p1, p2, pred1, pred2, y).detach(),
              "mscl": ((1 - 0.75) * sup(p1, p2, meta) + 0.75 * sup(pred1, pred2, y)).detach()}
    torch.save(losses, out / "losses.pt")
    print("losses", {k: round(float(v), 4) for k, v in losses.items() if v.dim() == 0})

    # ---- 5) all meta sa labels ----
    meta_rows = [{"filepath": r.filepath, "onset": float(r.onset), "offset": float(r.offset), "split": r.split,
                  "sa": int(r.sa_class_num), "sex": r.sex, "age_group": r.age_group} for r in df.itertuples()]
    json.dump(meta_rows, open(out / "meta_sa.json", "w"))
    print("meta rows", len(meta_rows), "done:", out)


if __name__ == "__main__":
    main()
