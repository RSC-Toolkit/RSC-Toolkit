"""
LoRA-RSC 5-seed reproduction check aggregation: outputs/lorarsc/<run>/best.json (gated on EXIT=0 in outputs/lorarsc/logs/<run>.done).

  .venv/bin/python scripts/reproduction/aggregate_lorarsc_results.py

Paper-reported value = repository README table (paper body not available; number of runs/± unknown, presumably a single run with the code default seed 10, best-on-test):
  LoRA(K,V r8 α8) Sp 85.31 / Se 36.11 / Score 60.71,  full-FT Sp 82.84 / Se 36.70 / Score 59.77.
"""
import json
import pathlib
import re
import statistics

PAPER = {
    "icbhi-lorarsc":        ("LoRA-RSC (K,V r=8 α=8, 0.299M) — official 60/40 split, seed 1–5", (85.31, 36.11, 60.71)),
    "icbhi-lorarsc-seed10": ("LoRA-RSC — code default seed 10 (compare with the README single run)", (85.31, 36.11, 60.71)),
    "icbhi-lorarsc-fullft": ("full fine-tuning baseline (lr 5e-5)", (82.84, 36.70, 59.77)),
    "icbhi-lorarsc-noise":  ("LoRA-RSC — 1D waveform-noise interpretation (per-sample N(0,0.3²), data=icbhi_lorarsc_noise; alternative interpretation)", (85.31, 36.11, 60.71)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/lorarsc")
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
        print(f"\n== {name} ({base}) — {len(rows)} run ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  README    : Sp {paper[0]:6.2f}  Se {paper[1]:6.2f}  Score {paper[2]:6.2f} (number of runs/± unknown)   Score diff: {sc[0]-paper[2]:+.2f}")
        else:
            print(f"  README    : Sp {paper[0]:6.2f}  Se {paper[1]:6.2f}  Score {paper[2]:6.2f} (number of runs/± unknown)   Score diff: {rows[0][3]-paper[2]:+.2f}")


if __name__ == "__main__":
    main()
