"""
BTS 5-seed reproduction check aggregation: collects outputs/bts/<run>/best.json (+ logs/<run>.done EXIT=0) and
prints the per-method seed mean±std (Sp/Se/Score) next to the paper-reported values.

  .venv/bin/python scripts/reproduction/aggregate_bts_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # BTS paper (2406.06786) Table 3 (5 seeds, ICBHI official 4-class)
    "icbhi-audioclap": ("Audio-CLAP",     (80.85, 3.33), (44.67, 3.77), (62.56, 0.37)),
    "icbhi-bts":       ("BTS (meta all)", (81.40, 2.57), (45.67, 2.66), (63.54, 0.80)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/bts")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            if not _completed(root, run):
                partial.append(run)
                continue
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            def ms(i):
                vals = [r[i] for r in rows]
                return statistics.mean(vals), statistics.stdev(vals)
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  paper     : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
            print(f"  Score diff: {sc[0]-psc[0]:+.2f} (paper ±{psc[1]:.2f})")


if __name__ == "__main__":
    main()
