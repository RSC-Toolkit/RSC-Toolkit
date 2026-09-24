"""
BTS-CARD 5-seed reproduction check aggregation: collects the ICBHI IND metrics from outputs/btscard/<run>/best.json (+ outputs/btscard/logs/<run>.done EXIT=0)
and the SPRSound OOD metrics from outputs/btscard/<run>/ood.json (produced by eval_btscard_checkpoint.py) as seed mean±std,
and prints them next to the paper-reported values of Table 2/3.

  .venv/bin/python scripts/reproduction/aggregate_btscard_results.py [--root outputs/btscard]

Protocol: toolkit main result = α=0 fixed · best on the ICBHI official test (highest Score & Se>5, best-on-test) · seeds 1–5 ·
SPRSound OOD = that ICBHI-best ckpt evaluated on the 2022 inter-patient 4-class merged set. The paper's Full is the best value over an α sweep (Table 3 footnote).
"""
import argparse
import json
import pathlib
import re
import statistics

PAPER = {  # (name, IND Sp/Se/Score, OOD Sp/Se/Score) — paper Table 2/3 (5 runs; unclear whether ± is variance or std)
    "icbhi-btscard":       ("BTS-CARD Full (code recipe 15ep adam)", (84.42, 3.47), (44.83, 2.94), (64.63, 0.57), (82.02, 3.28), (41.90, 4.96), (61.96, 1.50)),
    "icbhi-btscard-paper": ("BTS-CARD Full (paper recipe 30ep adamw)", (84.42, 3.47), (44.83, 2.94), (64.63, 0.57), (82.02, 3.28), (41.90, 4.96), (61.96, 1.50)),
    "icbhi-btscard-woa":   ("w/o (a) counterfactual", (83.49, None), (43.02, None), (63.25, None), (70.83, None), (46.27, None), (58.55, None)),
    "icbhi-btscard-wob":   ("w/o (b) adversarial", (82.84, None), (45.94, None), (64.33, None), (72.50, None), (46.27, None), (59.39, None)),
    "icbhi-btscard-woc":   ("w/o (c) meta augmentation", (83.55, None), (43.81, None), (63.68, None), (66.99, None), (47.73, None), (56.96, None)),
}
BASELINES = {"BTS (Table 2)": ((81.40, 45.67, 63.54), (67.50, 39.33, 53.42)),
             "Audio-CLAP (Table 2)": ((80.85, 44.67, 62.56), (70.67, 41.90, 56.29))}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _fmt(v):
    return f"{v[0]:6.2f}±{v[1]:.2f}" if v[1] is not None else f"{v[0]:6.2f}      "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/btscard")
    ap.add_argument("--ood_source", default="author", help="sprsound_<source> key in ood.json")
    ap.add_argument("--alpha", default="0.0")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base not in groups:
            continue
        if not _completed(root, run):
            partial.append(run)
            continue
        d = json.loads(best.read_text())
        ood = None
        oj = best.parent / "ood.json"
        if oj.exists():
            o = json.loads(oj.read_text()).get(f"sprsound_{args.ood_source}", {}).get(f"alpha={float(args.alpha):.1f}")
            if o:
                ood = (o["S_p"], o["S_e"], o["icbhi_score"])
        groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch"), ood))
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    print(f"protocol: α={args.alpha} fixed · best on the ICBHI official test (Se>5, best-on-test) · OOD = ICBHI-best ckpt → SPRSound 2022 inter 4-class ({args.ood_source})")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc, osp, ose, osc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep, ood in rows:
            o = f"  | OOD Sp {ood[0]:6.2f}  Se {ood[1]:6.2f}  Score {ood[2]:6.2f}" if ood else "  | OOD (no ood.json)"
            print(f"  seed {seed}: IND Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep}){o}")
        if len(rows) >= 2:
            def ms(vals):
                return statistics.mean(vals), statistics.stdev(vals)
            sp, se, sc = ms([r[1] for r in rows]), ms([r[2] for r in rows]), ms([r[3] for r in rows])
            print(f"  ours IND  : Sp {_fmt(sp)}  Se {_fmt(se)}  Score {_fmt(sc)}")
            print(f"  paper IND : Sp {_fmt(psp)}  Se {_fmt(pse)}  Score {_fmt(psc)}   Score diff {sc[0]-psc[0]:+.2f}")
            oods = [r[5] for r in rows if r[5]]
            if len(oods) >= 2:
                a, b, c = ms([o[0] for o in oods]), ms([o[1] for o in oods]), ms([o[2] for o in oods])
                print(f"  ours OOD  : Sp {_fmt(a)}  Se {_fmt(b)}  Score {_fmt(c)}  ({len(oods)} seeds)")
                print(f"  paper OOD : Sp {_fmt(osp)}  Se {_fmt(ose)}  Score {_fmt(osc)}   Score diff {c[0]-osc[0]:+.2f}")
    print("\nReference (paper Table 2 baselines): " + "; ".join(f"{k} IND {v[0]} / OOD {v[1]}" for k, v in BASELINES.items()))
    print("Toolkit BTS 5-seed run (outputs/bts/SUMMARY.txt): 63.91±0.87 (IND)")


if __name__ == "__main__":
    main()
