"""
SCL-metadata 5-seed reproduction check aggregation: outputs/sclmeta/<run>/best.json (+ logs/<run>.done EXIT=0) → paper Table 1·2 (CNN6 AudioSet).

  .venv/bin/python scripts/reproduction/aggregate_sclmeta_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # arXiv 2210.16192 Table 1/2, CNN6 AudioSet, 10 runs mean(std)
    "icbhi-sclmeta-ce":     ("CE",     (70.09, 3.08), (40.39, 2.97), (55.24, 0.43)),
    "icbhi-sclmeta-hybrid": ("Hybrid", (70.47, None), (43.29, None), (56.89, 0.55)),
    "icbhi-sclmeta-scl":    ("SCL",    (75.95, 2.31), (39.15, 1.89), (57.55, 0.81)),
    "icbhi-sclmeta-mscl":   ("M-SCL (sex+age & class)", (76.93, 2.99), (39.15, 2.84), (58.04, 0.94)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/sclmeta")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            if not _completed(root, run):
                partial.append(run); continue
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            def ms(i):
                vals = [r[i] for r in rows]
                return statistics.mean(vals), statistics.stdev(vals)
            sp, se, sc = ms(1), ms(2), ms(3)
            fmt = lambda v: f"{v[0]:6.2f}±{v[1]:.2f}" if v[1] is not None else f"{v[0]:6.2f}"
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  paper     : Sp {fmt(psp)}  Se {fmt(pse)}  Score {fmt(psc)} (10 runs)")
            print(f"  Score diff: {sc[0]-psc[0]:+.2f} (paper ±{psc[1]:.2f})")


if __name__ == "__main__":
    main()
