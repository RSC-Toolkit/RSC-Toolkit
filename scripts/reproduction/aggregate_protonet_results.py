"""
Prototype (Ren et al., ICASSP 2022) 5-seed reproduction check aggregation: outputs/protonet/<run>/best.json → compared with paper Table 3 (Test column).

Paper protocol: test evaluation of the **last iterate** at 10,000 iterations (no separate best selection) → `final_*` is the primary check,
and the toolkit default best-on-test is reported alongside for reference. The paper reports a single run (no ±).

  .venv/bin/python scripts/reproduction/aggregate_protonet_results.py
"""
import json
import pathlib
import re
import statistics

# paper Table 3 Test column (SE, SP, AS=Score)
PAPER = {
    "icbhi-protonet-cnn8": ("CNN-8 baseline (Table 3 row 1)", 39.42, 59.72, 49.57),
    "icbhi-protonet":      ("ProtoNet Pt-2D-EleSim-Van w/ BN (paper 'Ours')", 27.78, 72.96, 50.37),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/protonet")
    print("Prototype Learning for Interpretable Respiratory Sound Analysis (ICASSP 2022) — ICBHI official 60/40 split, 4-class")
    print("※ The paper reports a single run (no ±). The check uses the **last iterate (final_*)** because the official code evaluates only the 10,000-iteration model.")
    partial = []
    for base, (name, psе, psp, pas) in PAPER.items():
        rows = []
        for d in sorted(root.glob(f"{base}-s[0-9]*")):
            run = d.name
            if not re.fullmatch(rf"{re.escape(base)}-s\d+", run):
                continue
            bj = d / "best.json"
            if not bj.exists():
                continue
            if not _completed(root, run):
                partial.append(run)
                continue
            j = json.loads(bj.read_text())
            rows.append((
                int(run.rsplit("-s", 1)[1]),
                j.get("final_S_e"), j.get("final_S_p"), j.get("final_icbhi_score"),
                j.get("S_e"), j.get("S_p"), j.get("icbhi_score"),
            ))
        rows.sort()
        if not rows:
            continue
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for sd, fse, fsp, fsc, bse, bsp, bsc in rows:
            print(f"  seed {sd}: final Se {fse:6.2f} Sp {fsp:6.2f} Score {fsc:6.2f}  |  best-on-test Score {bsc:6.2f}")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            se, sp, sc = ms(1), ms(2), ms(3)
            bsc = ms(6)
            print(f"  ours mean (final)       : Se {se[0]:6.2f}±{se[1]:.2f}  Sp {sp[0]:6.2f}±{sp[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  ours mean (best-on-test): Score {bsc[0]:6.2f}±{bsc[1]:.2f} (reference)")
            print(f"  paper                   : Se {psе:6.2f}  Sp {psp:6.2f}  Score {pas:6.2f} (single run)   Score diff: {sc[0]-pas:+.2f}")
    if partial:
        print("\n(incomplete runs — excluded from aggregation: " + ", ".join(sorted(partial)) + ")")


if __name__ == "__main__":
    main()
