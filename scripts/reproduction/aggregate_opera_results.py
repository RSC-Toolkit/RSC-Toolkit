"""
OPERA-CT P1 집계: outputs/opera/<run>/best.json (+ logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_opera_results.py

앵커: EMBC'25(arXiv 2504.18004) OPERA-CT 동결 + Tfm 헤드 — ICBHI 4-class Score 48.97±1.21, SPRSound 7-class 71.62±0.84 (5회 반복).
전체 미세조정(ft) 은 툴킷 정의 레시피라 논문 목표치가 없다 (동결 대비 비교만).
"""
import json
import pathlib
import re
import statistics

PAPER = {
    "icbhi-operact-frozen": ("OPERA-CT 동결 + Tfm (ICBHI 4-class)", 48.97, 1.21),
    "sprs-operact-frozen":  ("OPERA-CT 동결 + Tfm (SPRSound 7-class)", 71.62, 0.84),
    "icbhi-operact-ft":     ("OPERA-CT 전체 미세조정 (ICBHI, 툴킷 레시피)", None, None),
    "sprs-operact-ft":      ("OPERA-CT 전체 미세조정 (SPRSound, 툴킷 레시피)", None, None),
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
        name, psc, pstd = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            if psc is not None:
                print(f"  EMBC 보고 : Score {psc:6.2f}±{pstd:.2f}   차이: {sc[0]-psc:+.2f}")
            else:
                print("  논문 목표치 없음 (툴킷 정의 레시피)")


if __name__ == "__main__":
    main()
