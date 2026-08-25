"""
M2D 미세조정 경로(nttcslab/m2d app/icbhi_sprs)의 P0 골든 — 공식 코드를 우리 venv 에서 실행 (nnAudio·einops 는 스크래치 pylib 를 PYTHONPATH 로).

  PYTHONPATH=<pylib> .venv/bin/python scripts/reproduction/dump_m2d_goldens.py --repo <m2d 클론> --ckpt <checkpoint-*.pth> --out tests/golden/m2d

골든:
- features.pt : tests/golden/sclmeta/audio.pt 파형(6개) → torchaudio mel 80(400/160, 50–8000) → dB (샘플별) ; val_transform (배치 min-max+Standardize 0.3671/0.2391) ;
                seed 고정 train_transform(SpecAug scl) 두 뷰
- encoder.pt  : 공식 RuntimeM2D(encoder_only, ckpt) 의 encode_lms(val 특징) → (6, 201, 3840) + state_dict 키/모양 + norm_stats + drop 키
- head.pt     : det_fill Linear(3840,4) 로짓 (mean pool) + 가중 CE
"""
import argparse
import json
import pathlib
import sys
import types

import torch


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sclmeta_repo", required=True)
    ap.add_argument("--out", default="tests/golden/m2d")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(args.repo).resolve()))
    sys.path.insert(0, str(pathlib.Path(args.sclmeta_repo).resolve()))
    sys.argv = [sys.argv[0]]
    from torchaudio import transforms as T
    from utils import Normalize, Standardize  # scl_icbhi2017
    from augmentations import SpecAugment
    from m2d.runtime_audio import RuntimeM2D, Config

    wav = torch.load("tests/golden/sclmeta/audio.pt")
    x = torch.stack([w["wav"] for w in wav]).unsqueeze(1)  # (6,1,128000)
    melspec = T.MelSpectrogram(n_fft=400, n_mels=80, win_length=400, hop_length=160, f_min=50, f_max=8000)
    normalize = Normalize(); standardize = Standardize(mean=0.3671, std=0.2391)
    specaug = SpecAugment(freq_mask=20, time_mask=40, freq_stripes=2, time_stripes=2)
    with torch.no_grad():
        dbmel = normalize.power_to_db(melspec(x))
        val = standardize(normalize(melspec(x)))
        torch.manual_seed(9)
        v1 = standardize(specaug(normalize(melspec(x))))
        v2 = standardize(specaug(normalize(melspec(x))))
    torch.save({"wav": x, "dbmel": dbmel, "val": val, "seed": 9, "view1": v1, "view2": v2}, out / "features.pt")
    print("features", tuple(dbmel.shape), tuple(val.shape))

    cfg = Config()
    rt = RuntimeM2D(cfg=cfg, weight_file=args.ckpt, training_mask=0.0, encoder_only=True, freeze_embed=True).eval()
    with torch.no_grad():
        emb = rt.encode_lms(val)
    sd = rt.backbone.state_dict()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    dropped = sorted(k for k in ckpt if k not in sd)
    torch.save({"emb": emb, "state_dict_shapes": {k: list(v.shape) for k, v in sd.items()},
                "norm_stats": rt.backbone.norm_stats.detach().clone(), "dropped": dropped,
                "pos_embed": sd["pos_embed"].clone(), "ckpt": args.ckpt}, out / "encoder.pt")
    print("encoder emb", tuple(emb.shape), "keys", len(sd), "dropped", len(dropped), "frozen:",
          all(not p.requires_grad for p in rt.backbone.patch_embed.parameters()))

    head = torch.nn.Linear(3840, 4)
    for _, p in sorted(head.named_parameters()):
        det_fill(p)
    logits = head(emb.mean(dim=1))
    w = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32); w = 1.0 / (w / w.sum()); w = w / w.sum()
    y = torch.tensor([wv["label"] for wv in wav])
    loss = torch.nn.functional.cross_entropy(logits, y, weight=w)
    torch.save({"logits": logits.detach(), "labels": y, "loss": loss.detach()}, out / "head.pt")
    print("head loss", float(loss), "완료:", out)


if __name__ == "__main__":
    main()
