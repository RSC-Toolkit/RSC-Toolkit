"""
Loads an MVST authors' released per-view checkpoint (best.pth) into the toolkit MVSTViewAST and evaluates the test set of the given split(s).
Checks whether files that reconstruct the authors' split (seed-1, depends on folder layout) reproduce the Table 1 single-view row (16×16: 78.48/46.62/62.55).

  .venv/bin/python scripts/reproduction/eval_mvst_checkpoint.py --view 16 --split <split.txt> [--split ...]
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from rsc_toolkit.models.ast_mvst import MVSTViewAST
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule


def build_dm(split_file):
    return IndexDataModule(
        dataset_name="icbhi", index_dataset_cls="rsc_toolkit.datasets.index_dataset.IndexDataset",
        project_root=str(pathlib.Path.cwd()), data_dir=".datasets/icbhi", split_file=split_file,
        sampling_rate=16000, sample_level="cycle", desired_length_sec=8.0, pad_type="repeat_fade",
        train_splits=["train"], val_splits=["test"], test_splits=["test"], batch_size=8, num_workers=8,
        features=dict(enabled=True, kind="fbank", cache_dir="downloaded_datasets/features_cache", n_mels=128,
                      target_frames=798, frame_shift=10, window_type="hanning", dither=0.0, normalize=True),
        feature_precompute=True, specaugment=dict(enabled=False),
    )


def score(preds, labels, n_cls=4):
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    sp = conf[0, 0] / conf[0].sum() * 100
    se = np.trace(conf[1:, 1:]) / conf[1:].sum() * 100
    return conf, sp, se, (sp + se) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=int, default=16)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = args.ckpt or f"pretrained_models/upstream/mvst/{args.view}/best.pth"
    m = MVSTViewAST(n_cls=4, view=args.view, audioset_pretrain=False)
    m.load_reference_checkpoint(ckpt)
    m.to(args.device).eval()
    results = {}
    for split in args.split:
        dm = build_dm(split); dm.setup("fit")
        preds, labels = [], []
        with torch.no_grad():
            for x, y in dm.val_dataloader():
                with torch.autocast(device_type="cuda", enabled=(args.amp and args.device.startswith("cuda"))):
                    logits = m(x.squeeze(1).to(args.device))
                preds.extend(logits.argmax(1).cpu().tolist()); labels.extend(y.tolist())
        conf, sp, se, sc = score(preds, labels)
        results[split] = {"n": len(labels), "S_p": sp, "S_e": se, "icbhi_score": sc, "conf": conf.tolist()}
        print(f"[view {args.view}] {pathlib.Path(split).name}: n={len(labels)}  Sp {sp:.2f} / Se {se:.2f} / Score {sc:.2f}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
