"""
SG-SCL 5-seed reproduction check aggregation: collects outputs/sgscl/<run>/best.json and
prints a table of the per-method seed mean±std (Sp/Se/Score) next to the paper-reported values.

  .venv/bin/python scripts/reproduction/aggregate_sgscl_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # SG-SCL paper (2312.09603) Table 3 (5 seeds, ICBHI official 4-class)
    "icbhi-ast-sgscl": ("SG-SCL (AST)",    (79.87, 8.89), (43.55, 5.93), (61.71, 1.61)),
    "icbhi-ast-dat":   ("DAT (AST, dev)",  (77.11, 7.20), (41.99, 5.00), (59.81, 1.25)),
    # non-AST — SG-SCL paper Table 3 (values re-run by the authors with the same repository recipe, bs64/400ep).
    # The values in Patch-Mix paper Table 1 (ResNet18 53.84±1.26, EffB0 56.56±0.79) come from different runs; reference only.
    "icbhi-resnet18-ce":       ("ResNet18 SFT",        (74.72, 3.43), (33.95, 3.88), (54.33, 0.91)),
    "icbhi-resnet18-sgscl":    ("ResNet18 + SG-SCL",   (75.58, 6.36), (34.63, 5.98), (55.10, 1.18)),
    "icbhi-effb0-ce":          ("EffNet-B0 SFT",       (73.48, 5.93), (39.24, 2.43), (57.46, 1.05)),
    "icbhi-effb0-sgscl":       ("EffNet-B0 + SG-SCL",  (81.58, 3.38), (33.85, 3.75), (57.72, 1.32)),
    "icbhi-cnn6-ce":           ("CNN6 SFT",            (80.13, 2.64), (35.91, 3.52), (58.10, 0.59)),
    "icbhi-cnn6-sgscl":        ("CNN6 + SG-SCL",       (78.16, 3.49), (38.05, 4.41), (58.11, 0.64)),
    "icbhi-resnet34-ce":       ("ResNet34 SFT",        None, None, None),
}

def _completed(root: pathlib.Path, run: str) -> bool:
    """
    A run counts as completed only if logs/<run>.done (EXIT=0), written by the launcher when training ends, exists.
    best.json is written on every best update, so runs that died midway also have one —
    they must not be aggregated as completed runs (e.g. when a Slurm job is terminated).
    """
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/sgscl")
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
            if psc is not None:
                print(f"  paper     : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
                print(f"  Score diff: {sc[0]-psc[0]:+.2f} (paper ±{psc[1]:.2f})")
            else:
                print("  paper     : (no paper value — reference only)")

if __name__ == "__main__":
    main()
