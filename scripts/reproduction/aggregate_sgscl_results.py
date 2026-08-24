"""
sgscl P1 결과 집계: outputs/sgscl/<run>/best.json 을 모아
방법별 seed 평균±표준편차(Sp/Se/Score)를 논문 보고치와 나란히 표로 찍는다.

  .venv/bin/python scripts/reproduction/aggregate_sgscl_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # SG-SCL 논문(2312.09603) Table 3 (5 seeds, ICBHI official 4-class)
    "icbhi-ast-sgscl": ("SG-SCL (AST)",    (79.87, 8.89), (43.55, 5.93), (61.71, 1.61)),
    "icbhi-ast-dat":   ("DAT (AST, dev)",  (77.11, 7.20), (41.99, 5.00), (59.81, 1.25)),
    # 비-AST (SFT 400ep 레시피; CE 행은 SG-SCL 논문 Table 3 자체 실행치)
    "icbhi-resnet18-ce":       ("ResNet18 SFT",        (76.63, 1.87), (31.06, 2.45), (53.84, 1.26)),
    "icbhi-resnet18-sgscl":    ("ResNet18 + SG-SCL",   (75.58, 6.36), (34.63, 5.98), (55.10, 1.18)),
    "icbhi-effb0-ce":          ("EffNet-B0 SFT",       (80.07, 4.74), (33.05, 4.14), (56.56, 0.79)),
    "icbhi-effb0-sgscl":       ("EffNet-B0 + SG-SCL",  (81.58, 3.38), (33.85, 3.75), (57.72, 1.32)),
    "icbhi-cnn6-ce":           ("CNN6 SFT",            (80.13, 2.64), (35.91, 3.52), (58.10, 0.59)),
    "icbhi-cnn6-sgscl":        ("CNN6 + SG-SCL",       (78.16, 3.49), (38.05, 4.41), (58.11, 0.64)),
    # ResNet34 SFT 는 두 논문 모두 자체 수치 없음 (Interspeech'23 의 55.20 은 인용) — 목표치 미설정
    "icbhi-resnet34-ce":       ("ResNet34 SFT",        None, None, None),
}

def main():
    root = pathlib.Path("outputs/sgscl")
    groups = {k: [] for k in PAPER}
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))

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
            if psc is not None:
                print(f"  논문 보고 : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f}")
                print(f"  Score 차이: {sc[0]-psc[0]:+.2f} (논문 ±{psc[1]:.2f})")
            else:
                print("  논문 보고 : (자체 수치 없음 — 참고용)")

if __name__ == "__main__":
    main()
