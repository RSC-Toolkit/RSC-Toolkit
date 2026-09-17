"""
DDE-MAE (Wei et al., Interspeech 2025) 5-seed reproduction check aggregation: outputs/ddemae/<run>/best.json (+ logs/<run>.done EXIT=0) → paper Table 1/2.
All runs are paper-spec re-implementations / assumption runs (no official code; only the Audio-MAE backbone is verified against goldens) — compared along the ladder B2 → S2 → S3.

  .venv/bin/python scripts/reproduction/aggregate_ddemae_results.py
"""
import json
import pathlib
import re
import statistics

# Paper Table 1/2 (± = 'Confidence Intervals' per the caption; confidence level and number of runs not reported)
PAPER = {
    "icbhi-ddemae-b2":      ("B2  MAE encoder + direct CE FT (Table 1)",            (64.47, 1.25), (50.72, 0.68), (57.59, 0.91)),
    "icbhi-ddemae-s2":      ("S2  reconstruction FT → classification (Table 1)",    (68.89, 1.32), (50.98, 0.76), (59.93, 0.79)),
    "icbhi-ddemae-s3":      ("S3  DDE-MAE disentangling FT → classification (Table 1, main result)", (69.32, 1.27), (53.69, 1.15), (61.50, 0.94)),
    "icbhi-ddemae-nomi":    ("ablation −MI (Table 2)",                           (69.11, 1.20), (53.50, 1.10), (61.30, 0.85)),
    "icbhi-ddemae-nosia":   ("ablation −Siamese (Table 2)",                      (68.54, 1.15), (51.74, 1.02), (60.14, 0.88)),
    "icbhi-ddemae-nomisia": ("ablation −(MI+Siamese) (Table 2 row 4, duplicated label → inferred configuration)", (67.61, 1.08), (48.71, 0.98), (58.16, 0.91)),
}
LADDER = ["icbhi-ddemae-b2", "icbhi-ddemae-s2", "icbhi-ddemae-s3"]


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/ddemae")
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
    print("DDE-MAE — paper-spec re-implementation / assumption runs (no public code; only the Audio-MAE backbone is verified against goldens). Assumptions: 4 kHz fbank 128 (21/128 empty filters)/1022+2 frames, waveform zero-padding (feature value −1.28, 57–91% of frames padded).")
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    summary = {}
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
            summary[base] = sc[0]
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f} (std)")
            print(f"  paper     : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f} (CI, number of runs not reported)")
            print(f"  Score diff: {sc[0]-psc[0]:+.2f}")
    if summary:
        print("\n== Ladder check (B2 → S2 → S3; if B2 does not match, the feature pipeline is at fault) ==")
        for base in LADDER:
            if base in summary:
                print(f"  {base}: ours {summary[base]:.2f} vs paper {PAPER[base][3][0]:.2f} ({summary[base]-PAPER[base][3][0]:+.2f})")


if __name__ == "__main__":
    main()
