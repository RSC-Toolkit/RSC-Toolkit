"""
AMS-D P1 집계: outputs/amsd/<run>/best.json (+ outputs/amsd/logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_amsd_results.py

판정 앵커 = **논문 이진 지표(비표준)** — 논문 §3.1: 4-class 학습, Se/Sp 는 비정상 전체=양성·정상=음성 이진 산식(공식 utils/metric.py).
표준 4-class ICBHI Score(정확한 비정상 클래스만 정답) 를 병기하되 Table 1 의 타 논문 행과는 비교 불가. run 수·± 미기재(단일 seed 2345 추정).
best 선택 = 툴킷 프로토콜(매 epoch 공식 test, 판정 키 최고; best.json 의 selection_key) — 코드의 train 내부 80/20 valid 선택과 다름.
앵커 대비 알려진 차이(집계 출력에도 표시): (1) dataset_aug(시간영역 증강, 클래스 불균형 보정) 없이 학습, (2) 클립 8 s loop-pad 는 가정(질문 #10),
(3) freqm/timem·mean/std 가정(질문 #3), (4) 공식은 (train+aug) 80% 서브셋 학습 + 20% valid best / test.py epoch 45.
"""
import json
import pathlib
import re
import statistics

# run 접두어 → (표기, 논문 이진 (Se, Sp, Score))
PAPER = {
    "icbhi-amsd":          ("AMS-D (attentive, full)      — Table 1/3", (60.92, 74.16, 67.54)),
    "icbhi-amsd-baseline": ("AST full-FT baseline         — Table 3 1행", (64.47, 67.15, 65.81)),
    "icbhi-amsd-sdonly":   ("+ Self-Distill (No Masking)  — Table 3 2행 / Table 2", (44.28, 82.79, 66.11)),
    "icbhi-amsd-sdmask":   ("+ Self-Distill + Mask        — Table 3 3행", (49.49, 74.35, 61.92)),
    "icbhi-amsd-random":   ("Random masking               — Table 2", (63.14, 70.68, 66.91)),
    "icbhi-amsd-time":     ("Time-Interval masking        — Table 2", (63.05, 67.64, 65.35)),
    "icbhi-amsd-freq":     ("Frequency-Interval masking   — Table 2", (89.42, 16.09, 52.75)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _ms(rows, key):
    vals = [r[key] for r in rows]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main():
    root = pathlib.Path("outputs/amsd")
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
        groups[base].append({"seed": seed, "epoch": d.get("epoch"), "sel": d.get("selection_key"),
                             "pSe": d.get("paper_S_e"), "pSp": d.get("paper_S_p"), "pSc": d.get("paper_score"),
                             "Se": d.get("S_e"), "Sp": d.get("S_p"), "Sc": d.get("icbhi_score")})
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
    print("※ '논문 지표(비표준)' = 이진 Se/Sp/Score (비정상 전체=양성). '표준' = 4-class ICBHI Score. 둘은 비교 불가.")
    print("※ 앵커 대비 알려진 차이: dataset_aug(불균형 보정 증강) 없음 | 8 s loop-pad·freqm/timem·mean/std 는 가정(저자 질문 #3,#10) | "
          "공식은 80% 서브셋+valid best(epoch 45), 툴킷은 train 100%+best-on-test")
    for base, rows in groups.items():
        if not rows:
            continue
        name, paper = PAPER[base]
        rows.sort(key=lambda r: r["seed"])
        print(f"\n== {name} ({base}) — {len(rows)} seeds, best 판정 키 {rows[0]['sel']} ==")
        for r in rows:
            print(f"  seed {r['seed']}: 논문 지표(비표준) Se {r['pSe']:6.2f} Sp {r['pSp']:6.2f} Score {r['pSc']:6.2f} | "
                  f"표준 4-class Sp {r['Sp']:6.2f} Se {r['Se']:6.2f} Score {r['Sc']:6.2f}  (epoch {r['epoch']})")
        pse, psp, psc = _ms(rows, "pSe"), _ms(rows, "pSp"), _ms(rows, "pSc")
        se, sp, sc = _ms(rows, "Se"), _ms(rows, "Sp"), _ms(rows, "Sc")
        print(f"  우리 평균 논문 지표(비표준): Se {pse[0]:6.2f}±{pse[1]:.2f}  Sp {psp[0]:6.2f}±{psp[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
        print(f"  우리 평균 표준 4-class     : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
        print(f"  논문 보고 논문 지표(비표준): Se {paper[0]:6.2f}  Sp {paper[1]:6.2f}  Score {paper[2]:6.2f} (run 수·± 미기재)   "
              f"Score 차이: {psc[0] - paper[2]:+.2f}")


if __name__ == "__main__":
    main()
