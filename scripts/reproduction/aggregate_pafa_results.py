"""
pafa P1 결과 집계: outputs/pafa/<run>/best.json (+ logs/<run>.done EXIT=0) → 논문 Table 1 대비.

  .venv/bin/python scripts/reproduction/aggregate_pafa_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # PAFA 논문(2505.23834) Table 1, ICBHI official 4-class, 5 seeds
    "icbhi-beats-ce":   ("BEATs + CE (SFT)", (78.77, 3.07), (48.21, 2.32), (63.49, 1.08)),
    "icbhi-beats-pafa": ("BEATs + PAFA",     (82.05, 1.95), (47.63, 2.23), (64.84, 0.60)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/pafa")
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
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            def ms(i):
                vals = [r[i] for r in rows]
                return statistics.mean(vals), statistics.stdev(vals)
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  논문 보고 : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
            print(f"  Score 차이: {sc[0]-psc[0]:+.2f} (논문 ±{psc[1]:.2f})")


if __name__ == "__main__":
    main()
