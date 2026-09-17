"""
MVST 5-seed reproduction check aggregation: outputs/mvst/icbhi-mvst-{official|seed1}-v{view}-s{seed}/best.json (views) and
icbhi-mvst-{split}-fusion-s{seed}/best.json (fusion) → compared with paper Table 1. Completion is decided by the per-chain .done (logs/icbhi-mvst-{split}-s{seed}.done).
Main result = official 60/40 split; seed1 (the paper's split, patient leakage) is for comparison.

  .venv/bin/python scripts/reproduction/aggregate_mvst_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # Table 1 (single run, the authors' seed-1 split)
    "v16": (78.48, 46.62, 62.55), "v32": (79.31, 42.61, 60.96), "v64": (84.55, 39.68, 62.11),
    "v128": (80.17, 45.71, 62.94), "v256": (84.63, 40.14, 62.38), "fusion": (81.99, 51.10, 66.55),
}


def main():
    root = pathlib.Path("outputs/mvst")
    for split in ("official", "seed1"):
        print(f"\n##### split {split} ({'toolkit main result' if split == 'official' else 'paper split, for comparison (patient leakage)'})")
        done_seeds = {int(m.group(1)) for p in (root / "logs").glob(f"icbhi-mvst-{split}-s*.done")
                      if "EXIT=0" in p.read_text() and (m := re.search(r"-s(\d+)\.done$", p.name))}
        for key in ("v16", "v32", "v64", "v128", "v256", "fusion"):
            rows = []
            for s in sorted(done_seeds):
                name = f"icbhi-mvst-{split}-{key}-s{s}" if key != "fusion" else f"icbhi-mvst-{split}-fusion-s{s}"
                bj = root / name / "best.json"
                if bj.exists():
                    d = json.loads(bj.read_text()); rows.append((s, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
            if not rows:
                continue
            psp, pse, psc = PAPER[key]
            print(f"== {key} — {len(rows)} seeds ==")
            for s, sp, se, sc, ep in rows:
                print(f"  seed {s}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
            if len(rows) >= 2:
                ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
                sp, se, sc = ms(1), ms(2), ms(3)
                print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  paper (Table 1, seed-1 split, single run): Sp {psp:6.2f}  Se {pse:6.2f}  Score {psc:6.2f}")


if __name__ == "__main__":
    main()
