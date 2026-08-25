"""
BTS 저자 체크포인트(best.pth)를 툴킷 BTSModel 에 그대로 로드해 공식 test 셋(test.pt)을
평가한다 — 구조·텍스트 파이프라인·데이터가 원본과 같다면 README 수치
(Sp 82.58 / Se 45.11 / Score 63.85) 가 나와야 한다 (P0 체크포인트 로드 동일성).

  .venv/bin/python scripts/reproduction/eval_bts_checkpoint.py [--device cuda:1] [--amp]

--amp: 원본 validate 처럼 autocast(fp16) 로 평가 (기본은 fp32).
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataset, bts_collate
from rsc_toolkit.models.bts import BTSModel


def get_score(hits, counts):
    """원본 util/icbhi_util.py get_score — Sp = 정상 정확도, Se = 비정상 정확도, Score = 평균."""
    sp = hits[0] / counts[0] * 100
    se = sum(hits[1:]) / sum(counts[1:]) * 100
    return sp, se, (sp + se) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pretrained_models/upstream/bts/best.pth")
    ap.add_argument("--test", default="pretrained_models/upstream/bts/test.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=None, help="결과 json 저장 경로")
    args = ap.parse_args()

    model = BTSModel(n_cls=4, meta_mode="all")
    model.load_reference_checkpoint(args.ckpt)
    model.to(args.device).eval()

    loader = DataLoader(BTSPrecomputedDataset(args.test), batch_size=args.batch_size,
                        shuffle=False, collate_fn=bts_collate, num_workers=4)
    hits, counts = [0.0] * 4, [0.0] * 4
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(args.device)
            with torch.autocast(device_type="cuda", enabled=(args.amp and args.device.startswith("cuda"))):
                logits = model(x, meta)
            preds = logits.argmax(1).cpu()
            for p, t in zip(preds.tolist(), y.tolist()):
                counts[t] += 1
                if p == t:
                    hits[t] += 1
    sp, se, sc = get_score(hits, counts)
    result = {"S_p": sp, "S_e": se, "icbhi_score": sc, "amp": args.amp, "n": int(sum(counts)),
              "reference_readme": {"S_p": 82.58, "S_e": 45.11, "icbhi_score": 63.85}}
    print(json.dumps(result, indent=1))
    print(f"Sp {sp:.2f} / Se {se:.2f} / Score {sc:.2f}  (README: 82.58 / 45.11 / 63.85)")
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
