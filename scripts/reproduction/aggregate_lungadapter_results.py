"""
LungAdapter 5-seed reproduction check aggregation: outputs/lungadapter/<run>/best.json (completion checked via EXIT=0 in logs/<run>.done when present).

  .venv/bin/python scripts/reproduction/aggregate_lungadapter_results.py

Paper-reported values (Interspeech 2024, 5 random runs): Table 1 4-class 80.43±3.11 / 44.37±2.46 / 62.40±1.11 (2.48M/90.01M).
Internal inconsistency in the paper: Figure 2 bars (recovered from PDF vectors) AST+LungAdapter = 75.53±2.83 / 45.21±2.48 / 60.37
→ both values are reported. Table 2 ablation: Attn only 77.72/45.05/61.39±1.75, Res only 77.55/44.55/61.05±0.51.
Fair in-toolkit comparison baseline: AST SFT (weighted CE, full FT) 5-seed 59.80±0.55 / paper 59.55.
"""
import json
import pathlib
import re
import statistics

PAPER = {  # run base name → (display name, Table values (Sp,Se,Score,Score std), Fig.2 recovered values or None)
    "icbhi-lungadapter":          ("LungAdapter (Attn+Res) — 4-class official 60/40 split", (80.43, 44.37, 62.40, 1.11), (75.53, 45.21, 60.37)),
    "icbhi-lungadapter-resscale": ("LungAdapter (Attn+Res, res_scale=vector: variant exactly matching the paper's 2.48M parameter count)", (80.43, 44.37, 62.40, 1.11), (75.53, 45.21, 60.37)),
    "icbhi-lungadapter-attnonly": ("ablation: Attention-Adapter only (Table 2)",  (77.72, 45.05, 61.39, 1.75), None),
    "icbhi-lungadapter-resonly":  ("ablation: Res-Adapter only (Table 2)",        (77.55, 44.55, 61.05, 0.51), None),
}


def _completed(root, run):
    # If logs/ does not exist (the launcher does not write .done), treat as completed — same convention as the addrsc worker. Incomplete runs may be mixed in.
    done = root / "logs" / f"{run}.done"
    return (not done.parent.exists()) or (done.exists() and "EXIT=0" in done.read_text())


def main():
    root = pathlib.Path("outputs/lungadapter")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        if run.startswith("smoke"):
            continue
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
        name, paper, fig2 = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean  : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  paper Table: Sp {paper[0]:6.2f}  Se {paper[1]:6.2f}  Score {paper[2]:6.2f}±{paper[3]:.2f}   Score diff {sc[0]-paper[2]:+.2f}")
            if fig2 is not None:
                print(f"  paper Fig.2: Sp {fig2[0]:6.2f}  Se {fig2[1]:6.2f}  Score {fig2[2]:6.2f} (recovered from PDF vectors, internally inconsistent)   Score diff {sc[0]-fig2[2]:+.2f}")
    print("\n(toolkit comparison baseline: AST SFT weighted-CE full FT, 5 seeds 59.80±0.55 — scripts/reproduction/aggregate_patchmix_results.py)")


if __name__ == "__main__":
    main()
