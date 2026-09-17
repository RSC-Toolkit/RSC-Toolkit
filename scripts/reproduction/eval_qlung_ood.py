"""
QLung SPRSound OOD evaluation — loads the ICBHI-trained best.pth into ast_qlung and evaluates zero-shot on the SPRSound 7→4 merged view (paper Table 2 OOD column).

  .venv/bin/python scripts/reproduction/eval_qlung_ood.py --run icbhi-qlung-s1            # outputs/qlung/<run>/best.pth → ood.json
  .venv/bin/python scripts/reproduction/eval_qlung_ood.py --ckpt <best.pth> --out <json>
  .venv/bin/python scripts/reproduction/eval_qlung_ood.py --run icbhi-qlung-clap-s1 --backbone clap   # Audio-CLAP variant (48 kHz CLAP OOD view)

The paper does not specify the evaluation scope, so both are aggregated: "inter" = 2022 inter-patient test 1,429, "all" = train + inter-test 8,085 (paper 3.1 distribution).
Preprocessing: configs/data/sprsound_qlung_ood.yaml (assumes the same chain as ICBHI). Metrics Sp/Se/Score = the formula in metrics/icbhi/score.py.
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
    ap.add_argument("--backbone", default="ast", choices=["ast", "clap"],
                    help="ast = ast_qlung + 16 kHz fbank OOD view | clap = clap_qlung (Audio-CLAP variant) + 48 kHz CLAP OOD view")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=0, help="for smoke tests")
    ap.add_argument("--threads", type=int, default=2, help="torch threads in the main process (64-thread STFT/resampling is 100x slower on a shared server)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers for the fbank cache warm-up")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.run:
        run_dir = pathlib.Path("outputs/qlung") / args.run
        args.ckpt = args.ckpt or str(run_dir / "best.pth")
        args.out = args.out or str(run_dir / "ood.json")
    if not args.ckpt:
        raise SystemExit("--run or --ckpt is required")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.backbone == "clap":
        m = ModelFactory.get_model_class("clap_qlung")(n_cls=4, method=args.method)
        data_config = "rsc_toolkit/configs/data/sprsound_qlung_clap_ood.yaml"
    else:
        m = ModelFactory.get_model_class("ast_qlung")(n_cls=4, method=args.method, input_layout="freq_time")
        data_config = "rsc_toolkit/configs/data/sprsound_qlung_ood.yaml"
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    real_missing = [k for k in missing if not k.startswith("dfam.class_margin")]
    assert not real_missing and not unexpected, (real_missing, unexpected)
    m.to(args.device).eval()
    amp = not args.no_amp and args.device.startswith("cuda")

    results = {"ckpt": args.ckpt, "backbone": args.backbone, "icbhi_best": {k: ck.get(k) for k in ("epoch", "S_p", "S_e", "icbhi_score")}}
    for scope in args.scopes.split(","):
        splits = SCOPES[scope]
        dm = build_sprsound_ood_datamodule(splits, data_config=data_config, overrides={"qlung": {"precompute": False},
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
        print(f"[{scope}] n={len(labels)} (set {sum(counts)} = {counts})  Sp {sp:.2f} / Se {se:.2f} / Score {sc:.2f}")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print("→", args.out)


if __name__ == "__main__":
    main()
