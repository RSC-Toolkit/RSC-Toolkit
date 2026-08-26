"""
QLung SPRSound OOD 평가 — ICBHI 학습 best.pth 를 ast_qlung 에 로드해 SPRSound 7→4 병합 뷰를 제로샷 평가 (논문 Table 2 OOD 열).

  .venv/bin/python scripts/reproduction/eval_qlung_ood.py --run icbhi-qlung-s1            # outputs/qlung/<run>/best.pth → ood.json
  .venv/bin/python scripts/reproduction/eval_qlung_ood.py --ckpt <best.pth> --out <json>

평가 범위가 논문에 없어 둘 다 집계한다: "inter" = 2022 inter-patient test 1,429, "all" = train + inter-test 8,085 (논문 3.1 분포).
전처리는 configs/data/sprsound_qlung_ood.yaml (ICBHI 와 같은 체인 가정). 지표 Sp/Se/Score = metrics/icbhi/score.py 산식.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from rsc_toolkit.datasets.sprsound_ood_fbank import build_sprsound_ood_datamodule, class_counts_4
from rsc_toolkit.models.factory import ModelFactory

SCOPES = {"inter": ["test"], "all": ["train", "test"]}


def score(preds, labels, n_cls=4):
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    sp = conf[0, 0] / max(conf[0].sum(), 1) * 100
    se = np.trace(conf[1:, 1:]) / max(conf[1:].sum(), 1) * 100
    return conf, sp, se, (sp + se) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="outputs/qlung/<run> (best.pth, ood.json)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scopes", default="inter,all")
    ap.add_argument("--method", default="aqm")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=0, help="스모크용")
    ap.add_argument("--threads", type=int, default=2, help="메인 프로세스 torch 스레드 (공유 서버에서 64 스레드 STFT/리샘플은 100배 느리다)")
    ap.add_argument("--workers", type=int, default=8, help="fbank 캐시 웜업 DataLoader 워커")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.run:
        run_dir = pathlib.Path("outputs/qlung") / args.run
        args.ckpt = args.ckpt or str(run_dir / "best.pth")
        args.out = args.out or str(run_dir / "ood.json")
    if not args.ckpt:
        raise SystemExit("--run 또는 --ckpt 필요")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = ModelFactory.get_model_class("ast_qlung")(n_cls=4, method=args.method, input_layout="freq_time")
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    real_missing = [k for k in missing if not k.startswith("dfam.class_margin")]
    assert not real_missing and not unexpected, (real_missing, unexpected)
    m.to(args.device).eval()
    amp = not args.no_amp and args.device.startswith("cuda")

    results = {"ckpt": args.ckpt, "icbhi_best": {k: ck.get(k) for k in ("epoch", "S_p", "S_e", "icbhi_score")}}
    for scope in args.scopes.split(","):
        splits = SCOPES[scope]
        dm = build_sprsound_ood_datamodule(splits, overrides={"qlung": {"precompute": False},
                                                             "feature_precompute_num_workers": args.workers})
        dm.setup("test")
        counts = class_counts_4(dm._build_index_dataset(splits=splits))
        preds, labels = [], []
        t0 = time.time()
        with torch.no_grad():
            for bi, batch in enumerate(dm.test_dataloader()):
                if args.limit_batches and bi >= args.limit_batches:
                    break
                x, y = batch[0], batch[1]
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits = m(x.squeeze(1).to(args.device))
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        conf, sp, se, sc = score(preds, labels)
        results[scope] = {"splits": splits, "n": len(labels), "class_counts": counts, "S_p": sp, "S_e": se,
                          "icbhi_score": sc, "conf": conf.tolist(), "seconds": time.time() - t0}
        print(f"[{scope}] n={len(labels)} (셋 {sum(counts)} = {counts})  Sp {sp:.2f} / Se {se:.2f} / Score {sc:.2f}")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print("→", args.out)


if __name__ == "__main__":
    main()
