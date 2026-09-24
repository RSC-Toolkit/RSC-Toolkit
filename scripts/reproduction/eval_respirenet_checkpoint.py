"""
Evaluates the RespireNet authors' ckpt_best.pkl with the toolkit RespireNetCNN + RespireNetChunkDataset (fold-4 test,
cycle majority vote). With the same pad_seed/res_type it must match tests/golden/respirenet/ckpt_eval.json down to the chunk predictions.

  .venv/bin/python scripts/reproduction/eval_respirenet_checkpoint.py [--split official]
"""
import argparse
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from rsc_toolkit.datasets.respirenet_chunks import RespireNetChunkDataset, respirenet_collate
from rsc_toolkit.models.respirenet import RespireNetCNN


def vote(preds, identifiers, cycle_ids):
    preds, ids = np.asarray(preds), np.asarray(identifiers)
    y_pred, y_true = [], []
    for cid in cycle_ids:
        y_pred.append(int(np.argmax(np.bincount(preds[np.where(ids == cid)]))))
        y_true.append(int(cid.split("_")[-1]))
    conf = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    sp = conf[0, 0] / conf[0].sum() * 100
    se = np.trace(conf[1:, 1:]) / conf[1:].sum() * 100
    return conf, sp, se, (sp + se) / 2, float((np.array(y_true) == np.array(y_pred)).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("RSCTK_RESPIRENET_REF_CKPT", "outputs/reference/respirenet/models/ckpt_best.pkl"),
                    help="the authors' fine-tuned ckpt (microsoft/RespireNet models/ckpt_best.pkl); defaults to the copy in the authors' code checkout, override with RSCTK_RESPIRENET_REF_CKPT")
    ap.add_argument("--data", default=".datasets/icbhi")
    ap.add_argument("--split", default="fold", choices=["fold", "official"])
    ap.add_argument("--test_fold", type=int, default=4)
    ap.add_argument("--pad_seed", type=int, default=0)
    ap.add_argument("--res_type", default="kaiser_best")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--golden", default="tests/golden/respirenet/ckpt_eval.json")
    args = ap.parse_args()
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"authors' ckpt not found: {args.ckpt} — place models/ckpt_best.pkl from microsoft/RespireNet there or specify it with --ckpt / RSCTK_RESPIRENET_REF_CKPT")

    ds = RespireNetChunkDataset(args.data, train_flag=False, split_mode=args.split, test_fold=args.test_fold,
                                folds_file="rsc_toolkit/datasets/icbhi/data/patient_list_foldwise.txt",
                                official_split_file="rsc_toolkit/datasets/icbhi/data/official_split.txt",
                                res_type=args.res_type, pad_seed=args.pad_seed)
    m = RespireNetCNN(n_cls=4, imagenet_pretrain=False)
    m.load_reference_checkpoint(args.ckpt)
    m.to(args.device).eval()
    preds = []
    with torch.no_grad():
        for x, y, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=respirenet_collate):
            preds.extend(m(x.to(args.device)).argmax(1).cpu().tolist())
    conf, sp, se, sc, acc = vote(preds, ds.identifiers, ds.cycle_ids)
    print(f"[{args.split}] chunks {len(ds)} cycles {len(ds.cycle_ids)}  Sp {sp:.2f} / Se {se:.2f} / Score {sc:.2f} / acc {acc:.2f}")
    print(conf)
    if args.split == "fold" and args.test_fold == 4:
        try:
            g = json.load(open(args.golden))
            same = int(np.sum(np.array(g["chunk_preds"]) == np.array(preds)))
            print(f"chunk predictions matching the golden {same}/{len(preds)} | golden Score {g['icbhi_score']:.2f} conf identical={g['conf']==conf.tolist()}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
