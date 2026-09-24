"""
BTS-CARD checkpoint evaluation — ICBHI official test (IND) α sweep + SPRSound 2022 inter-patient 4-class OOD.

  # authors' released ckpt (compared with README/Table 2; a single ckpt, so only closeness is checked)
  .venv/bin/python scripts/reproduction/eval_btscard_checkpoint.py --ckpt pretrained_models/upstream/btscard/best.pth \
      --ckpt_format author --sprsound author,toolkit --amp --device cuda:2 --out outputs/btscard/eval_author_ckpt.json
  # toolkit training output (ICBHIBestScoreCheckpoint best.pth) → OOD evaluation (chained with && after training in the launcher queue)
  .venv/bin/python scripts/reproduction/eval_btscard_checkpoint.py --ckpt outputs/btscard/icbhi-btscard-s1/best.pth \
      --ckpt_format toolkit --sprsound author --out outputs/btscard/icbhi-btscard-s1/ood.json

Output json: (S_p, S_e, icbhi_score) for icbhi[α] / sprsound_<source>[α] + standalone scores of TE(o_multi) and NDE(o_text) (in the α=0 row,
same as the original validate log). --amp autocasts the whole pass (fp16) like the original validate.
Protocol: the toolkit main result reports IND/OOD from one ckpt (α=0 fixed, best on the ICBHI test with Se>5); the α sweep is reported alongside for reference.
"""
import argparse
import json
import pathlib

import torch
from torch.utils.data import DataLoader

from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataset, bts_collate
from rsc_toolkit.datasets.sprsound_ood import build_ood_dataset, describe_meta3_sprsound, sprsound_ood_collate
from rsc_toolkit.models.bts_card import BTSCARDModel

ALPHAS = [round(0.1 * i, 1) for i in range(11)]
README = {"icbhi": {"S_p": 84.42, "S_e": 44.83, "icbhi_score": 64.63},
          "sprsound": {"S_p": 82.02, "S_e": 41.90, "icbhi_score": 61.96},
          "note": "paper Table 2 (5-run mean; Full = best value over the α sweep). The authors' best.pth records its own best_acc inside the ckpt"}


def get_score(hits, counts):
    sp = hits[0] / (counts[0] + 1e-10) * 100
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    return {"S_p": sp, "S_e": se, "icbhi_score": (sp + se) / 2}


def _count(preds, y, hits, counts):
    for p, t in zip(preds.tolist(), y.tolist()):
        counts[t] += 1
        if p == t:
            hits[t] += 1


@torch.no_grad()
def evaluate(model, loader, device, amp, describe):
    hits = {a: [0.0] * 4 for a in ALPHAS}
    counts = {a: [0.0] * 4 for a in ALPHAS}
    h_te, c_te, h_nde, c_nde = [0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4
    for batch in loader:
        x, y, meta = batch[0], batch[1], batch[2]
        x = x.to(device)
        texts = describe(meta)
        with torch.autocast(device_type="cuda", enabled=(amp and device.startswith("cuda"))):
            out = model.forward_all(x, texts, texts)
        o1, o2 = out["o1"].float(), out["o2"].float()
        _count(out["o_multi"].argmax(1).cpu(), y, h_te, c_te)
        _count(out["o_text"].argmax(1).cpu(), y, h_nde, c_nde)
        for a in ALPHAS:
            _count((o1 - a * o2).argmax(1).cpu(), y, hits[a], counts[a])
    res = {f"alpha={a:.1f}": get_score(hits[a], counts[a]) for a in ALPHAS}
    res["TE(o_multi)"] = get_score(h_te, c_te)
    res["NDE(o_text)"] = get_score(h_nde, c_nde)
    res["n"] = int(sum(c_te))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_format", choices=["author", "toolkit"], default="toolkit")
    ap.add_argument("--icbhi_test", default="pretrained_models/upstream/bts/test.pt")
    ap.add_argument("--no_icbhi", action="store_true")
    ap.add_argument("--sprsound", default="author", help="comma-separated: author, toolkit, none")
    ap.add_argument("--rhonchi_stridor", default="drop", choices=["drop"], help="4-class view: only dropping Rhonchi/Stridor is allowed (the inter-test set has none)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--mask_mode", default="omit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = BTSCARDModel(n_cls=4, mask_mode=args.mask_mode)
    if args.ckpt_format == "author":
        info = model.load_reference_checkpoint(args.ckpt)
    else:
        model.load_toolkit_checkpoint(args.ckpt)
        side = pathlib.Path(args.ckpt).with_suffix(".json")
        info = json.load(open(side)) if side.exists() else {}
    model.to(args.device).eval()

    result = {"ckpt": args.ckpt, "ckpt_format": args.ckpt_format, "ckpt_info": info, "amp": args.amp,
              "protocol": "toolkit main result: α=0 fixed, IND/OOD evaluated with the best ckpt on the ICBHI official test (Se>5); the α sweep is for reference",
              "reference_table2": README}
    if not args.no_icbhi:
        loader = DataLoader(BTSPrecomputedDataset(args.icbhi_test), batch_size=args.batch_size, shuffle=False,
                            collate_fn=bts_collate, num_workers=2)
        result["icbhi"] = evaluate(model, loader, args.device, args.amp, model.describe)
        print("ICBHI test:", json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in result["icbhi"].items() if isinstance(v, dict)}, indent=0))
    for src in [s.strip() for s in args.sprsound.split(",") if s.strip() and s.strip() != "none"]:
        ds = build_ood_dataset(src, rhonchi_stridor=args.rhonchi_stridor)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=sprsound_ood_collate, num_workers=2)
        result[f"sprsound_{src}"] = evaluate(model, loader, args.device, args.amp,
                                             lambda meta: [describe_meta3_sprsound(m) for m in meta])
        print(f"SPRSound OOD ({src}):", json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in result[f"sprsound_{src}"].items() if isinstance(v, dict)}, indent=0))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1, ensure_ascii=False)
        print("saved", args.out)


if __name__ == "__main__":
    main()
