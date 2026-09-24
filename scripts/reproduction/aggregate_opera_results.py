"""
OPERA-CT 5-seed reproduction check aggregation: outputs/opera/<run>/best.json (+ logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_opera_results.py

Paper-reported values: EMBC'25 (arXiv 2504.18004) frozen OPERA-CT + Tfm head — ICBHI 4-class Score 48.97±1.21, SPRSound 7-class 71.62±0.84 (5 repetitions).
Full fine-tuning (ft) uses a toolkit-defined recipe, so it has no paper-reported value (compared only against the frozen setting).
"""
import json
import pathlib
import re
import statistics

PAPER = {
    # EMBC'25 Table I / II OPERA-CT rows: (Sp, Se, Score) mean±std
    "icbhi-operact-frozen": ("OPERA-CT frozen + Tfm (ICBHI 4-class)", ((79.09, 9.51), (18.86, 8.95), (48.97, 1.21))),
    "sprs-operact-frozen":  ("OPERA-CT frozen + Tfm (SPRSound 7-class)", ((83.23, 4.72), (60.00, 5.58), (71.62, 0.84))),
    "icbhi-operact-ft":     ("OPERA-CT full fine-tuning (ICBHI, toolkit recipe)", None),
    "sprs-operact-ft":      ("OPERA-CT full fine-tuning (SPRSound, toolkit recipe)", None),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/opera")
    groups = {k: [] for k in PAPER}
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups and _completed(root, run):
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
    for base, rows in groups.items():
        if not rows:
            continue
        name, paper = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            if paper is not None:
                (psp, psps), (pse, pses), (psc, pscs) = paper
                print(f"  EMBC      : Sp {psp:6.2f}±{psps:.2f}  Se {pse:6.2f}±{pses:.2f}  Score {psc:6.2f}±{pscs:.2f}   Score diff: {sc[0]-psc:+.2f}")
            else:
                print("  no paper-reported value (toolkit-defined recipe)")


if __name__ == "__main__":
    main()
