"""
BTS-CARD 체크포인트 평가 — ICBHI 공식 test(IND) α 스윕 + SPRSound 2022 inter-patient 4-class OOD.

  # 저자 배포 ckpt (README/Table 2 대조; 단일 ckpt 라 근접 여부만)
  .venv/bin/python scripts/reproduction/eval_btscard_checkpoint.py --ckpt pretrained_models/upstream/btscard/best.pth \
      --ckpt_format author --sprsound author,toolkit --amp --device cuda:2 --out outputs/btscard/eval_author_ckpt.json
  # 툴킷 학습 산출물 (ICBHIBestScoreCheckpoint best.pth) → OOD 평가 (런처 큐에서 학습 뒤 && 로 연결)
  .venv/bin/python scripts/reproduction/eval_btscard_checkpoint.py --ckpt outputs/btscard/icbhi-btscard-s1/best.pth \
      --ckpt_format toolkit --sprsound author --out outputs/btscard/icbhi-btscard-s1/ood.json

출력 json: icbhi[α] / sprsound_<source>[α] 의 (S_p, S_e, icbhi_score) + TE(o_multi)·NDE(o_text) 단독 점수 (α=0 행에서,
원본 validate 로그와 동일). --amp 는 원본 validate 처럼 전 과정 autocast(fp16).
프로토콜 표기: 툴킷 주 결과는 α=0 고정·ICHBI test 기준 best(Se>5) ckpt 하나로 IND/OOD 를 낸다; α 스윕은 참고용 병기.
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
          "note": "논문 Table 2 (5 runs mean; Full = α 스윕 중 최고값). 저자 best.pth 자체 best_acc 는 ckpt 안에 기록"}


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
    ap.add_argument("--sprsound", default="author", help="쉼표 구분: author, toolkit, none")
    ap.add_argument("--rhonchi_stridor", default="drop", choices=["drop"], help="4-class 뷰: Rhonchi/Stridor 제외만 허용 (2026-09-16 결정; inter-test 엔 0 건)")
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
              "protocol": "툴킷 주 결과: α=0 고정, ICBHI 공식 test 기준 best(Se>5) ckpt 로 IND/OOD 평가; α 스윕은 참고",
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
