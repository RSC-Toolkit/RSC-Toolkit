"""
ADD-RSC 5-seed reproduction check aggregation: outputs/addrsc/<run>/best.json (+ logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_addrsc_results.py

The paper (Interspeech 2025, Table 1) AST variant 85.13/45.94/65.53 is judged to be a **single seed 0, code split (random.Random(1) file-level 60/40)**
result (the paper states the official 60/40 split). Main result = official 60/40 split (toolkit purpose); comparison = code-split reconstruction.
"""
import json
import pathlib
import re
import statistics

PAPER = {
    "icbhi-addrsc":      ("ADD-RSC(AST) — official 60/40 split (main result)", None),
    "icbhi-addrsc-code": ("ADD-RSC(AST) — code split (random.Random(1) file-level, for comparison)", (85.13, 45.94, 65.53)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/addrsc")
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
                print(f"  paper     : Sp {paper[0]:6.2f}  Se {paper[1]:6.2f}  Score {paper[2]:6.2f} (single seed 0)   Score diff: {sc[0]-paper[2]:+.2f}")
            else:
                print("  no paper value (the paper states the official split, but the code uses a random file split — compare with the code-split result)")


if __name__ == "__main__":
    main()
