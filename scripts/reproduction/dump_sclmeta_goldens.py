"""
SCL-metadata(ilyassmoummad/scl_icbhi2017) P0 골든 — 공식 코드를 우리 venv 에서 실행 (torch/torchaudio 만 필요).

  .venv/bin/python scripts/reproduction/dump_sclmeta_goldens.py --repo <클론> --data .datasets/icbhi --out tests/golden/sclmeta

골든:
- audio.pt        : metadata.csv 의 train 사이클 몇 개(짧은 것·8초 초과 포함)를 dataset.py 경로(구간 로드→리샘플→fade→순환→fade_out)로 → 128000 샘플
- features.pt     : 위 파형 배치 → 샘플별 dB-mel(캐시 단계), val_transform 출력(배치 min-max+Standardize), seed 고정 train_transform 두 뷰
- cnn6.pt         : PANN 부분 로드 CNN6 의 state_dict 모양 + val 특징 → 512-d 임베딩
- losses.pt       : 고정 proj/라벨/메타로 SupCon(labels/None)·SupConCE(α .5, 가중치)·M-SCL 합성 스칼라
- meta_sa.json    : metadata.csv 전 행의 (filepath, onset, offset) → sa_class_num (툴킷 meta 'sa' 대조용)
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
    # args.py 는 import 시 argparse 를 파싱한다 — 기본값으로 파싱되게 argv 를 비운다
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

    # ---- 1) 오디오: dataset.py get_sample + 패딩 경로를 몇 행에 대해 실행 ----
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

    # ---- 2) 특징 ----
    melspec = T.MelSpectrogram(n_fft=1024, n_mels=64, win_length=1024, hop_length=512, f_min=50, f_max=2000)
    normalize = Normalize()
    standardize = Standardize()
    specaug = SpecAugment(freq_mask=20, time_mask=40, freq_stripes=2, time_stripes=2)
    train_transform = torch.nn.Sequential(torch.nn.Sequential(melspec, normalize), specaug, standardize)
    val_transform = torch.nn.Sequential(torch.nn.Sequential(melspec, normalize), standardize)
    x = data  # (N, 1, 128000)
    with torch.no_grad():
        dbmel = normalize.power_to_db(melspec(x))          # 샘플별 캐시 단계
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

    # ---- 4) 손실 ----
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

    # ---- 5) meta sa 라벨 전수 ----
    meta_rows = [{"filepath": r.filepath, "onset": float(r.onset), "offset": float(r.offset), "split": r.split,
                  "sa": int(r.sa_class_num), "sex": r.sex, "age_group": r.age_group} for r in df.itertuples()]
    json.dump(meta_rows, open(out / "meta_sa.json", "w"))
    print("meta rows", len(meta_rows), "완료:", out)


if __name__ == "__main__":
    main()
