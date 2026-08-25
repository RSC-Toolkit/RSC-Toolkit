"""
M2D 미세조정 P1 집계: outputs/m2d/<run>/best.json (+ logs/<run>.done EXIT=0) → TASLP Table XIII.

  .venv/bin/python scripts/reproduction/aggregate_m2d_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # TASLP(2404.06095) Table XIII, ICBHI 4-class, 4 사전학습 시드 x 5 미세조정 = 20 평균 ±95% CI
    "icbhi-m2d-base": ("M2D/0.7 (16x4, 230529 공개 ckpt)",      (79.48, 1.47), (44.38, 1.55), (61.93, 0.25)),
    "icbhi-m2dx-s5":  ("M2D-X/0.7 η=0.3 (공개 ckpt = seed 5 만)", (81.51, 1.03), (45.08, 1.10), (63.29, 0.22)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/m2d")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            if not _completed(root, run):
                partial.append(run); continue
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
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
            print(f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  논문 보고 : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f} (20 run, 95% CI)")
            print(f"  Score 차이: {sc[0]-psc[0]:+.2f}")


if __name__ == "__main__":
    main()
