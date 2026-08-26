"""
Lung-SRAD P1 집계: outputs/lungsrad/<run>/best.json (+ logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_lungsrad_results.py

앵커(논문 Table 1/2, 공식 60/40, 5 seeds mean±std, best-on-test Se>5): 사다리 FT → SAR → Lung-SRAD 순으로 대조.
4-class(S_p/S_e/icbhi_score) 와 2-class 파생(paper_S_p/paper_S_e/paper_score; 4-class 가중치, preds>0→abnormal) 병기.
Dual-Axis Patch-Mix 의 축 정의는 **as_code 기본(원본 코드 동작: 'time'=주파수 32bin, 'freq'=채널 96) · as_paper 병행(Eq.6-7)** — 논문 수치를
낳은 판본은 저자 확인 전 불명이므로 두 변형을 나란히 표기한다.
"""
import json
import pathlib
import re
import statistics

# run 이름 접두 → (표시 이름, 4-class 앵커 (Sp, Se, Score, ±Score), 2-class 앵커 (Sp, Se, Score, ±Score))
PAPER = {
    "icbhi-lungsrad-ft":        ("(1) DASS FT (method=ce)",                                  (74.68, 47.43, 61.06, 1.27), (73.74, 61.72, 68.20, 2.05)),
    "icbhi-lungsrad-sar":       ("(2) + Spectral-Aware Regularization (method=ce_blur)",     (76.72, 47.72, 62.22, 1.29), (76.72, 60.29, 68.40, 1.04)),
    "icbhi-lungsrad-dual":      ("(6) Lung-SRAD = SAR + Dual-Axis Patch-Mix CL [as_code 기본]", (79.53, 49.42, 64.48, 0.25), (79.53, 65.61, 72.57, 0.47)),
    "icbhi-lungsrad-dualpaper": ("(6') Lung-SRAD [as_paper 병행: Eq.6-7 시간/주파수 축]",   (79.53, 49.42, 64.48, 0.25), (79.53, 65.61, 72.57, 0.47)),
    # Table 2 변형 (준비만; 큐 투입은 사다리 완료 후)
    "icbhi-lungsrad-2d":        ("(3) (2) + AST-style Patch-Mix CL (cl_mode=2d) [as_code]",   (74.39, 50.46, 62.42, 0.33), None),
    "icbhi-lungsrad-freq":      ("(4) (2) + Freq-only Patch-Mix CL (cl_mode=freq) [as_code]", (76.43, 49.94, 63.18, 1.31), None),
    "icbhi-lungsrad-time":      ("(5) (2) + Time-only Patch-Mix CL (cl_mode=time) [as_code]", (76.68, 48.97, 62.83, 0.69), None),
    "icbhi-lungsrad-2dpaper":   ("(3') AST-style Patch-Mix CL [as_paper]",                   (74.39, 50.46, 62.42, 0.33), None),
    "icbhi-lungsrad-freqpaper": ("(4') Freq-only [as_paper]",                                (76.43, 49.94, 63.18, 1.31), None),
    "icbhi-lungsrad-timepaper": ("(5') Time-only [as_paper]",                                (76.68, 48.97, 62.83, 0.69), None),
}
ORDER = list(PAPER)


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _ms(vals):
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main():
    root = pathlib.Path("outputs/lungsrad")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base not in groups:
            continue
        if not _completed(root, run):
            partial.append(run)
            continue
        d = json.loads(best.read_text())
        groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("paper_S_p"), d.get("paper_S_e"), d.get("paper_score"), d.get("epoch")))
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
    print("축 정의: as_code 기본(원본 코드 동작) · as_paper 병행(Eq.6-7). 4-class = 표준 ICBHIScore, 2-class = 4-class 가중치 파생(preds>0→abnormal).")
    summary = []
    for base in ORDER:
        rows = groups[base]
        if not rows:
            continue
        name, p4, p2 = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, sp, se, sc, bsp, bse, bsc, ep in rows:
            two = f"  | 2-class Sp {bsp:6.2f} Se {bse:6.2f} Score {bsc:6.2f}" if bsc is not None else ""
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep}){two}")
        sp, se, sc = _ms([r[1] for r in rows]), _ms([r[2] for r in rows]), _ms([r[3] for r in rows])
        print(f"  우리 4-class : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
        print(f"  논문 4-class : Sp {p4[0]:6.2f}        Se {p4[1]:6.2f}        Score {p4[2]:6.2f}±{p4[3]:.2f}   Score 차이: {sc[0]-p4[2]:+.2f}")
        line = {"base": base, "n": len(rows), "score4": sc, "paper4": p4[2]}
        if all(r[6] is not None for r in rows):
            bsp, bse, bsc = _ms([r[4] for r in rows]), _ms([r[5] for r in rows]), _ms([r[6] for r in rows])
            print(f"  우리 2-class : Sp {bsp[0]:6.2f}±{bsp[1]:.2f}  Se {bse[0]:6.2f}±{bse[1]:.2f}  Score {bsc[0]:6.2f}±{bsc[1]:.2f}")
            if p2 is not None:
                print(f"  논문 2-class : Sp {p2[0]:6.2f}        Se {p2[1]:6.2f}        Score {p2[2]:6.2f}±{p2[3]:.2f}   Score 차이: {bsc[0]-p2[2]:+.2f}")
            line["score2"] = bsc
        summary.append(line)
    if summary:
        print("\n== 사다리 요약 (4-class Score, 우리 vs 논문) ==")
        for s in summary:
            print(f"  {s['base']:<28} n={s['n']}  {s['score4'][0]:6.2f}±{s['score4'][1]:.2f}  vs  {s['paper4']:6.2f}")


if __name__ == "__main__":
    main()
