"""
RespireNet 5-seed reproduction check aggregation: outputs/respirenet/<run>/best.json (+ logs/<run>.done EXIT=0).
best.json holds both the Sp/Se/Score under the original rule (highest test cycle accuracy) and the Score-best entry for reference (best_score_info).

  .venv/bin/python scripts/reproduction/aggregate_respirenet_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # README Quantitative Results = paper Table 1 (presumably a single run). CNN / CNN+CBA+BRC
    "icbhi-respirenet-fold4":    ("RespireNet CNN(+BRC) 80/20 fold-4", (78.8, None), (53.6, None), (66.2, None), "CNN+CBA+BRC 67.1 · re-evaluation of the authors' ckpt 66.14"),
    "icbhi-respirenet-official": ("RespireNet CNN(+BRC) official 60/40 split",  (71.4, None), (39.0, None), (55.2, None), "CNN+CBA+BRC 55.7 (protocol assumption: toolkit official_split)"),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/respirenet")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            if not _completed(root, run):
                partial.append(run); continue
            d = json.loads(best.read_text()); b = d.get("best_score_info") or {}
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch"), d.get("cycle_acc"),
                                 b.get("icbhi_score"), b.get("epoch")))
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc, note = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep, acc, bsc, bep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  acc {acc:6.2f} (epoch {ep})  | Score-best {bsc:6.2f} (epoch {bep})")
        if len(rows) >= 2:
            def ms(i):
                vals = [r[i] for r in rows]
                return statistics.mean(vals), statistics.stdev(vals)
            sp, se, sc, bsc = ms(1), ms(2), ms(3), ms(6)
            print(f"  ours mean (acc-best, original rule): Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  ours mean (Score-best, reference)  : Score {bsc[0]:6.2f}±{bsc[1]:.2f}")
            print(f"  paper     : Sp {psp[0]:6.2f}  Se {pse[0]:6.2f}  Score {psc[0]:6.2f}  ({note})")
            print(f"  Score diff: {sc[0]-psc[0]:+.2f}")


if __name__ == "__main__":
    main()
