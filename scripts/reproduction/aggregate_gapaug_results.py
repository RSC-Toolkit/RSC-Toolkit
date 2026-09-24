"""
GaP-Aug (ICASSP 2024) 5-seed reproduction check aggregation — **paper-spec re-implementation / assumption runs** (no official or unofficial code, grade C).

  .venv/bin/python scripts/reproduction/aggregate_gapaug_results.py [--root outputs/gapaug]

Reads: outputs/gapaug/<run>/best.json (best-on-test S_p/S_e/icbhi_score/epoch + final_* last iterate) and EXIT=0 in logs/<run>.done.
Paper-reported values (official 60/40 split, single values; number of runs/± not reported): Table 1 GaP-aug+Mixup(CNN14) Se 58.20 / Sp 77.07 / Score 67.64;
Table 2 Naïve 48.34/64.28/56.31, GaP-aug w/o Mixup 56.49/76.94/66.72, SpecAug w/ Mixup 48.63/79.54/64.09,
PatchMask w/ Mixup 54.88/77.01/65.94, Mixup 55.88/71.82/63.85, SpecAug w/o 50.89/77.96/64.43, PatchMask w/o 54.88/76.18/65.53.
Check: 5-seed mean vs 67.64; secondary = relative gain over Naïve (paper +10.41 w/o mixup, +11.33 w/ mixup).
"""
import argparse
import json
import pathlib
import re
import statistics

# run prefix → (display name, paper Se, Sp, Score)
PAPER = {
    "icbhi-gapaug":           ("GaP-aug w/ Mixup (Table 1/2)", 58.20, 77.07, 67.64),
    "icbhi-gapaug-nomixup":   ("GaP-aug w/o Mixup",            56.49, 76.94, 66.72),
    "icbhi-gapaug-naive":     ("Naïve (no augmentation)",      48.34, 64.28, 56.31),
    "icbhi-gapaug-specaug":   ("SpecAug w/ Mixup",             48.63, 79.54, 64.09),
    "icbhi-gapaug-patchmask": ("PatchMask w/ Mixup",           54.88, 77.01, 65.94),
    "icbhi-gapaug-mixup":     ("Mixup only",                   55.88, 71.82, 63.85),
    "icbhi-gapaug-specaug-nomixup":   ("SpecAug w/o Mixup",   50.89, 77.96, 64.43),
    "icbhi-gapaug-patchmask-nomixup": ("PatchMask w/o Mixup", 54.88, 76.18, 65.53),
    # deviation-flag arm: w_trip=0.01 (λ of the sister paper, JMIR AI 2025) — paper Algorithm 1 uses an unweighted sum (default w_trip=1.0)
    "icbhi-gapaug-wtrip001":  ("GaP-aug w/ Mixup, w_trip=0.01 (deviation flag)", 58.20, 77.07, 67.64),
}
NAIVE = "icbhi-gapaug-naive"


def _completed(root: pathlib.Path, run: str) -> bool:
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _ms(vals):
    if not vals:
        return (float("nan"), float("nan"))
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/gapaug")
    ap.add_argument("--include-partial", action="store_true", help="also include runs without .done")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        if run.startswith("smoke"):
            continue
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base not in groups:
            continue
        if not _completed(root, run) and not args.include_partial:
            partial.append(run); continue
        d = json.loads(best.read_text())
        groups[base].append(dict(seed=seed, sp=d["S_p"], se=d["S_e"], sc=d["icbhi_score"], ep=d.get("epoch"),
                                 fsp=d.get("final_S_p"), fse=d.get("final_S_e"), fsc=d.get("final_icbhi_score"),
                                 facc=d.get("final_acc@1")))
    print("== GaP-Aug (Chang et al., ICASSP 2024) — paper-spec re-implementation / assumption runs (no code; assumptions: min-max non-negativization, Beta α=1, "
          "triplet weight 1.0@emb (Algo. 1 unweighted sum; 0.01 is the deviation arm), head_input=emb, hann window, 2 forward passes, no fade, front truncation, etc.) ==")
    print("   Table 1 secondary paper-reported Acc 69.01 is compared only against final_acc@1 (last iterate); best entries have no acc.")
    print("   Protocol: official 60/40 split, best-on-test (every 500 steps, no Se gate) + last iterate (final) alongside. The paper reports single values (number of runs/± not reported).")
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    means = {}
    for base, rows in groups.items():
        if not rows:
            continue
        name, pse, psp, psc = PAPER[base]
        rows.sort(key=lambda r: r["seed"])
        print(f"\n-- {name} ({base}) — {len(rows)} seed --")
        for r in rows:
            fin = f"  | final Sp {r['fsp']:.2f} Se {r['fse']:.2f} Sc {r['fsc']:.2f}" if r["fsc"] is not None else ""
            if r["facc"] is not None:
                fin += f" Acc {r['facc']:.2f}"
            print(f"  seed {r['seed']}: best Sp {r['sp']:6.2f}  Se {r['se']:6.2f}  Score {r['sc']:6.2f} (epoch {r['ep']}){fin}")
        sp, se, sc = _ms([r["sp"] for r in rows]), _ms([r["se"] for r in rows]), _ms([r["sc"] for r in rows])
        means[base] = dict(sp=sp, se=se, sc=sc, n=len(rows))
        print(f"  ours mean (best) : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
        if all(r["fsc"] is not None for r in rows):
            fsc = _ms([r["fsc"] for r in rows]); fse = _ms([r["fse"] for r in rows]); fsp = _ms([r["fsp"] for r in rows])
            means[base]["fsc"] = fsc
            print(f"  ours mean (final): Sp {fsp[0]:6.2f}±{fsp[1]:.2f}  Se {fse[0]:6.2f}±{fse[1]:.2f}  Score {fsc[0]:6.2f}±{fsc[1]:.2f}")
        if all(r["facc"] is not None for r in rows):
            facc = _ms([r["facc"] for r in rows])
            print(f"  ours mean (final Acc): {facc[0]:6.2f}±{facc[1]:.2f}" + ("  (paper Table 1 Acc 69.01)" if base == "icbhi-gapaug" else ""))
        print(f"  paper            : Sp {psp:6.2f}       Se {pse:6.2f}       Score {psc:6.2f} (single value)")
        print(f"  Score diff (best): {sc[0]-psc:+.2f}")
    if NAIVE in means:
        n_sc = means[NAIVE]["sc"][0]
        print("\n-- Gain over Naïve (secondary check) --")
        for base, mv in means.items():
            if base == NAIVE:
                continue
            gain = mv["sc"][0] - n_sc
            paper_gain = PAPER[base][3] - PAPER[NAIVE][3]
            print(f"  {PAPER[base][0]:<28}: ours {gain:+.2f}  vs paper {paper_gain:+.2f}")


if __name__ == "__main__":
    main()
