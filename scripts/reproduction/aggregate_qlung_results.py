"""
QLung 5-seed reproduction check aggregation: collects outputs/qlung/<run>/best.json (+ ood.json) and prints the per-AQS-variant seed mean±std next to the paper-reported values of Table 1/2/3.

  .venv/bin/python scripts/reproduction/aggregate_qlung_results.py

Completion: EXIT=0 in outputs/qlung/logs/<run>.done (written by the launcher worker after training && OOD evaluation). A run with only best.json is incomplete.
"""
import json
import pathlib
import re
import statistics

# (display name, paper Sp, Se, Score) — QLung paper (2606.11915) Table 1/3 (ICBHI official 60/40 split, seed 1–5)
PAPER = {
    "icbhi-qlung":         ("QLung on AST — code as-is (AQS rms bug)",   (81.90, 5.38), (42.12, 3.94), (62.01, 1.18)),
    "icbhi-qlung-entropynanmean": ("QLung on AST — code formula + entropy (nanmean, toolkit extension; approximates Figure 1)", (81.90, 5.38), (42.12, 3.94), (62.01, 1.18)),
    "icbhi-qlung-paper":   ("QLung on AST — paper Eq. (5) as-is (reference)",     (81.90, 5.38), (42.12, 3.94), (62.01, 1.18)),
    "icbhi-qlung-ce":      ("CE path (code quirk; Table 3 CE 77.14/41.97/59.55)", (77.14, 3.35), (41.97, 2.84), (59.55, 1.92)),
}
PAPER_OOD = {"QLung on AST": ((82.48, 8.35), (33.99, 8.21), (58.23, 3.83))}
ABLATION = "Table 3 (no flags; listed for reference only): +Fixed margin 75.65/43.77/59.72±0.89 · +Audio quality 80.99/39.95/60.47±0.96 · +Class imbalance 79.16/41.95/60.56±0.83"


def _completed(root: pathlib.Path, run: str) -> bool:
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _ms(vals):
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    root = pathlib.Path("outputs/qlung")
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
        if not _completed(root, run):
            partial.append(run)
            continue
        d = json.loads(best.read_text())
        ood_p = best.parent / "ood.json"
        ood = json.loads(ood_p.read_text()) if ood_p.exists() else None
        groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch"), ood))
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")

    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep, ood in rows:
            line = f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})"
            if ood:
                for scope in ("inter", "all"):
                    if scope in ood:
                        o = ood[scope]
                        line += f"  | OOD {scope} {o['S_p']:.2f}/{o['S_e']:.2f}/{o['icbhi_score']:.2f}"
            print(line)
        if len(rows) >= 2:
            sp, se, sc = _ms([r[1] for r in rows]), _ms([r[2] for r in rows]), _ms([r[3] for r in rows])
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  paper     : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
            print(f"  Score diff: {sc[0]-psc[0]:+.2f} (paper ±{psc[1]:.2f})")
            for scope in ("inter", "all"):
                oo = [r[5][scope] for r in rows if r[5] and scope in r[5]]
                if len(oo) >= 2:
                    osp, ose, osc = _ms([o["S_p"] for o in oo]), _ms([o["S_e"] for o in oo]), _ms([o["icbhi_score"] for o in oo])
                    n = oo[0]["n"]
                    print(f"  OOD {scope:5s}(n={n}): Sp {osp[0]:6.2f}±{osp[1]:.2f}  Se {ose[0]:6.2f}±{ose[1]:.2f}  Score {osc[0]:6.2f}±{osc[1]:.2f}"
                          f"   (paper Table 2 OOD QLung-AST 82.48±8.35 / 33.99±8.21 / 58.23±3.83, scope not specified)")
    print("\n" + ABLATION)
    print("Audio-CLAP variant (63.39 / OOD 59.80): no code — not implemented (documented only).")


if __name__ == "__main__":
    main()
