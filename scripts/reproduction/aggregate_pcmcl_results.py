"""
PC-MCL 5-seed reproduction check aggregation: outputs/pcmcl/<run>/best.json (gated on EXIT=0 in logs/<run>.done) → compared with paper Table 1/2/3.
The hyperparameters are **assumptions (PAFA lineage)** because the authors' scripts are not available, so every row is labelled as an assumption run.

  .venv/bin/python scripts/reproduction/aggregate_pcmcl_results.py
"""
import json
import pathlib
import re
import statistics

# arXiv 2601.17080 Table 1/2/3 — ICBHI official 60/40 4-class, 5 seeds mean±std (Sp, Se, Score)
PAPER = {
    "icbhi-pcmcl-beats":              ("BEATs + PC-MCL (Table 1/2, main result)",    (79.04, 1.90), (51.71, 2.98), (65.37, 0.73)),
    "icbhi-pcmcl-ast":                ("AST + PC-MCL (Table 1/2)",                   (78.54, 1.87), (46.05, 1.62), (62.30, 0.50)),
    # The two CE baseline rows only register the paper-reported values — there is no matching config/queue entry (input length and epochs unconfirmed). Their labels say "not run" until they are run.
    "icbhi-pcmcl-beats-ce":           ("BEATs + CE baseline (Table 2; setup not specified, not run)", (76.85, 1.88), (48.79, 1.72), (62.82, 0.62)),
    "icbhi-pcmcl-ast-ce":             ("AST + CE baseline (Table 2; setup not specified, not run)",   (77.14, 5.43), (41.97, 5.04), (59.55, 0.50)),
    "icbhi-pcmcl-beats-multionly":    ("BEATs ablation Multi only (Table 3)",        (73.10, 2.47), (51.10, 2.97), (62.10, 0.64)),
    "icbhi-pcmcl-beats-concatsingle": ("BEATs ablation Concat only (Table 3)",       (78.05, 2.49), (50.33, 2.93), (64.19, 0.31)),
    "icbhi-pcmcl-beats-nopm":         ("BEATs ablation Concat+Multi, no PM (Table 3)", (76.66, 2.93), (51.91, 2.25), (64.28, 0.62)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/pcmcl")
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
    print("※ All runs are assumption runs (no training script → PAFA-lineage assumptions: 100ep/bs32/Adam 5e-5/cosine/MA 0.5/nospec) — check withheld until the authors confirm")
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) [assumption run] — {len(rows)}/5 seeds ==")
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
