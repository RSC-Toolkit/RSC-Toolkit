"""
BTS-CARD SPRSound OOD input comparison — the authors' released valid_spr.pt vs the toolkit's own pipeline (configs/data/sprsound_btscard.yaml).

  .venv/bin/python scripts/reproduction/check_btscard_sprsound_parity.py [--out tests/golden/btscard/sprsound_parity.json]

Compared: (1) per-recording event count and order, (2) 4-class label agreement, (3) meta3 agreement, (4) text agreement,
(5) log-mel feature max|diff| (precedent from the ICBHI comparison tests/golden/bts/pipeline_parity_test_split.json: 7.6e-6).
On the first run the toolkit computes and caches the CLAP log-mel features of the 1,429 events (a few minutes on CPU).
"""
import argparse
import collections
import json
import pathlib

import torch

from rsc_toolkit.datasets.sprsound_ood import (
    SPRSoundOODAuthorDataset,
    build_toolkit_ood_dataset,
    describe_meta3_sprsound,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="pretrained_models/upstream/btscard/valid_spr.pt")
    ap.add_argument("--out", default="tests/golden/btscard/sprsound_parity.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    author = SPRSoundOODAuthorDataset(args.author)
    toolkit = build_toolkit_ood_dataset()
    by_rec_a = collections.defaultdict(list)
    for i in range(len(author)):
        feat, y, meta, rid = author[i]
        by_rec_a[rid].append((i, y, meta))
    by_rec_t = collections.defaultdict(list)
    for j, (idx, y4, meta3, sid, label7) in enumerate(toolkit.rows):
        rid = sid.rsplit("_cycle_", 1)[0]
        by_rec_t[rid].append((j, y4, meta3, sid))

    rec_a, rec_t = set(by_rec_a), set(by_rec_t)
    res = {"n_author": len(author), "n_toolkit": len(toolkit), "recordings_author": len(rec_a),
           "recordings_toolkit": len(rec_t), "recordings_only_author": sorted(rec_a - rec_t)[:10],
           "recordings_only_toolkit": sorted(rec_t - rec_a)[:10], "count_mismatch_recordings": [],
           "label_mismatch": 0, "meta_mismatch": 0, "text_mismatch": 0, "n_compared": 0,
           "max_abs_diff": 0.0, "n_diff_gt_1e-5": 0, "n_diff_gt_1e-3": 0, "first_bad": [],
           "class_counts_author": [0, 0, 0, 0], "class_counts_toolkit": toolkit.class_counts()}
    for i in range(len(author)):
        res["class_counts_author"][author[i][1]] += 1
    n = 0
    for rid in sorted(rec_a & rec_t):
        la, lt = by_rec_a[rid], by_rec_t[rid]
        if len(la) != len(lt):
            res["count_mismatch_recordings"].append([rid, len(la), len(lt)])
            continue
        for (ia, ya, ma), (jt, yt, mt, sid) in zip(la, lt):
            if args.limit and n >= args.limit:
                break
            fa = author[ia][0].float()
            ft = toolkit[jt][0].float()
            if ya != yt:
                res["label_mismatch"] += 1
            if not torch.allclose(ma, torch.tensor(mt)):
                res["meta_mismatch"] += 1
            if describe_meta3_sprsound(ma) != describe_meta3_sprsound(mt):
                res["text_mismatch"] += 1
            d = float((fa - ft).abs().max()) if fa.shape == ft.shape else float("inf")
            res["max_abs_diff"] = max(res["max_abs_diff"], d)
            if d > 1e-5:
                res["n_diff_gt_1e-5"] += 1
            if d > 1e-3:
                res["n_diff_gt_1e-3"] += 1
                if len(res["first_bad"]) < 10:
                    res["first_bad"].append([sid, d, list(fa.shape), list(ft.shape)])
            n += 1
    res["n_compared"] = n
    res["author_example"] = {"meta3": author[0][2].tolist(), "text": describe_meta3_sprsound(author[0][2]),
                             "recording": author[0][3]}
    print(json.dumps(res, indent=1))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("EXIT=0")


if __name__ == "__main__":
    main()
