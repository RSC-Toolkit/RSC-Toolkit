"""
MVST 3단계: 뷰 5개의 best.pth 로 특징을 덤프하고 게이티드 융합(FuseMLP)을 학습·평가한다 (공식 save_features.py + fusion.py).

  .venv/bin/python scripts/reproduction/mvst_fusion.py --ckpt-dir pretrained_models/upstream/mvst \
      --split rsc_toolkit/datasets/icbhi/data/mvst_seed1_split.txt --out outputs/mvst/fusion-author-seed1

--ckpt-dir 아래 {16,32,64,128,256}/best.pth (저자 배포) 또는 --ckpts view=path ... (툴킷 학습 best.pth: {"model": state_dict}).
특징 덤프: train 로더(셔플·drop_last·SpecAugment 샘플별) 1 pass 에서 같은 배치를 5 뷰에 모두 통과(원본은 seed 로 정렬을 보장) / test 는 순서대로.
"""
import argparse
import json
import pathlib

import torch

from rsc_toolkit.methods.mvst_fusion import train_fusion
from rsc_toolkit.models.ast_mvst import MVSTViewAST
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

VIEWS = (16, 32, 64, 128, 256)


def build_dm(split_file, seed):
    return IndexDataModule(
        dataset_name="icbhi", index_dataset_cls="rsc_toolkit.datasets.index_dataset.IndexDataset",
        project_root=str(pathlib.Path.cwd()), data_dir=".datasets/icbhi", split_file=split_file,
        sampling_rate=16000, sample_level="cycle", desired_length_sec=8.0, pad_type="repeat_fade",
        train_splits=["train"], val_splits=["test"], test_splits=["test"], batch_size=8, num_workers=8,
        shuffle_train=True, drop_last_train=True,
        features=dict(enabled=True, kind="fbank", cache_dir="downloaded_datasets/features_cache", n_mels=128,
                      target_frames=798, frame_shift=10, window_type="hanning", dither=0.0, normalize=True),
        feature_precompute=True, specaugment=dict(enabled=True, policy="icbhi_ast_sup", mask="mean"),
    )


def load_view(view, path, device):
    m = MVSTViewAST(n_cls=4, view=view, audioset_pretrain=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "classifier" in ckpt:      # 저자 save_model 형식
        m.load_reference_checkpoint(path)
    else:                          # 툴킷 ICBHIBestScoreCheckpoint 형식 {"model": state_dict}
        m.load_state_dict(ckpt["model"], strict=True)
    return m.to(device).eval()


@torch.no_grad()
def dump(models, loader, device, amp):
    feats = {v: [] for v in models}; labels = []
    for x, y in loader:
        x = x.squeeze(1).to(device)
        with torch.autocast(device_type="cuda", enabled=amp):
            for v, m in models.items():
                feats[v].append(m.forward_features(m.to_model_input(x)).float().cpu())
        labels.append(y)
    return [torch.cat(feats[v]) for v in VIEWS], torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpts", nargs="*", default=[], help="view=path ...")
    ap.add_argument("--split", default="rsc_toolkit/datasets/icbhi/data/official_split.txt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    paths = {v: f"{args.ckpt_dir}/{v}/best.pth" for v in VIEWS} if args.ckpt_dir else {}
    for kv in args.ckpts:
        v, p = kv.split("=", 1); paths[int(v)] = p
    assert set(paths) == set(VIEWS), paths
    import lightning as L
    L.seed_everything(args.seed, workers=True)  # 원본 main.py 의 seed (셔플·SpecAug 마스크)
    models = {v: load_view(v, paths[v], args.device) for v in VIEWS}
    dm = build_dm(args.split, args.seed); dm.setup("fit")
    amp = args.device.startswith("cuda")
    tr_feats, tr_labels = dump(models, dm.train_dataloader(), args.device, amp)
    te_feats, te_labels = dump(models, dm.val_dataloader(), args.device, amp)
    print(f"특징 덤프: train {tuple(tr_feats[0].shape)} test {tuple(te_feats[0].shape)}")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"train": tr_feats, "train_labels": tr_labels, "test": te_feats, "test_labels": te_labels}, out / "features.pt")
    best = train_fusion(tr_feats, tr_labels, te_feats, te_labels, seed=args.seed, device=args.device, amp=amp)
    torch.save({"model": best.pop("state_dict"), **best}, out / "best.pth")
    json.dump({**best, "split": args.split, "ckpts": paths}, open(out / "best.json", "w"), indent=1)
    print(f"MVST 융합 best: Sp {best['S_p']:.2f} / Se {best['S_e']:.2f} / Score {best['icbhi_score']:.2f} (epoch {best['epoch']}) → {out}")


if __name__ == "__main__":
    main()
