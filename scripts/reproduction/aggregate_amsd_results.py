"""
AMS-D 5-seed reproduction check aggregation: outputs/amsd/<run>/best.json (+ outputs/amsd/logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_amsd_results.py

Reference = **the paper's binary metric (non-standard)** — paper §3.1: 4-class training; Se/Sp use a binary formula with all abnormal = positive, normal = negative (official utils/metric.py).
The standard 4-class ICBHI Score (only the exact abnormal class counts as correct) is reported alongside, but is not comparable with the other papers' rows in Table 1. Number of runs and ± not reported (presumably a single seed 2345).
best selection = toolkit protocol (official test every epoch, highest selection key; selection_key in best.json) — differs from the code's selection on an internal 80/20 train/valid split.
Known differences from the paper-reported value (also shown in the output): (1) trained without dataset_aug (time-domain augmentation that corrects class imbalance), (2) the 8 s loop-pad clip is an assumption,
(3) freqm/timem and mean/std are assumptions, (4) the official code trains on an 80% subset of (train+aug) + 20% valid best / test.py epoch 45.
"""
import json
import pathlib
import re
import statistics

# run prefix → (label, paper binary (Se, Sp, Score))
PAPER = {
    "icbhi-amsd":          ("AMS-D (attentive, full)      — Table 1/3", (60.92, 74.16, 67.54)),
    "icbhi-amsd-baseline": ("AST full-FT baseline         — Table 3 row 1", (64.47, 67.15, 65.81)),
    "icbhi-amsd-sdonly":   ("+ Self-Distill (No Masking)  — Table 3 row 2 / Table 2", (44.28, 82.79, 66.11)),
    "icbhi-amsd-sdmask":   ("+ Self-Distill + Mask        — Table 3 row 3", (49.49, 74.35, 61.92)),
    "icbhi-amsd-random":   ("Random masking               — Table 2", (63.14, 70.68, 66.91)),
    "icbhi-amsd-time":     ("Time-Interval masking        — Table 2", (63.05, 67.64, 65.35)),
    "icbhi-amsd-freq":     ("Frequency-Interval masking   — Table 2", (89.42, 16.09, 52.75)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _ms(rows, key):
    vals = [r[key] for r in rows]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main():
    root = pathlib.Path("outputs/amsd")
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
        groups[base].append({"seed": seed, "epoch": d.get("epoch"), "sel": d.get("selection_key"),
                             "pSe": d.get("paper_S_e"), "pSp": d.get("paper_S_p"), "pSc": d.get("paper_score"),
                             "Se": d.get("S_e"), "Sp": d.get("S_p"), "Sc": d.get("icbhi_score")})
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    print("※ 'paper metric (non-standard)' = binary Se/Sp/Score (all abnormal = positive). 'standard' = 4-class ICBHI Score. The two are not comparable.")
    print("※ Known differences from the paper: no dataset_aug (imbalance-correcting augmentation) | 8 s loop-pad, freqm/timem, mean/std are assumptions | "
          "official: 80% subset + valid best (epoch 45); toolkit: 100% train + best-on-test")
    for base, rows in groups.items():
        if not rows:
            continue
        name, paper = PAPER[base]
        rows.sort(key=lambda r: r["seed"])
        print(f"\n== {name} ({base}) — {len(rows)} seeds, best selection key {rows[0]['sel']} ==")
        for r in rows:
            print(f"  seed {r['seed']}: paper metric (non-standard) Se {r['pSe']:6.2f} Sp {r['pSp']:6.2f} Score {r['pSc']:6.2f} | "
                  f"standard 4-class Sp {r['Sp']:6.2f} Se {r['Se']:6.2f} Score {r['Sc']:6.2f}  (epoch {r['epoch']})")
        pse, psp, psc = _ms(rows, "pSe"), _ms(rows, "pSp"), _ms(rows, "pSc")
        se, sp, sc = _ms(rows, "Se"), _ms(rows, "Sp"), _ms(rows, "Sc")
        print(f"  ours mean, paper metric (non-standard): Se {pse[0]:6.2f}±{pse[1]:.2f}  Sp {psp[0]:6.2f}±{psp[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
        print(f"  ours mean, standard 4-class           : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
        print(f"  paper, paper metric (non-standard)    : Se {paper[0]:6.2f}  Sp {paper[1]:6.2f}  Score {paper[2]:6.2f} (number of runs and ± not reported)   "
              f"Score diff: {psc[0] - paper[2]:+.2f}")


if __name__ == "__main__":
    main()
