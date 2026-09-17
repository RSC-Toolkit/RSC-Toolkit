"""
Patch-Mix 5-seed reproduction check aggregation: collects outputs/patchmix/<run>/best.json and
prints a table of the per-method seed mean±std (Sp/Se/Score) next to the paper-reported values.

  .venv/bin/python scripts/reproduction/aggregate_patchmix_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # paper Table 1·2 (5 seeds)
    "icbhi-ast-sft":     ("AST SFT",      (77.14, 3.35), (41.97, 2.21), (59.55, 0.88)),
    "icbhi-patchmix-ce": ("Patch-Mix CE", (76.87, 5.05), (42.04, 3.98), (59.46, 0.78)),
    "icbhi-patchmix-cl": ("Patch-Mix CL", (81.66, 3.83), (43.07, 2.80), (62.37, 0.61)),
}

def main():
    root = pathlib.Path("outputs/patchmix")
    groups = {k: [] for k in PAPER}
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))

    for base, rows in groups.items():
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
