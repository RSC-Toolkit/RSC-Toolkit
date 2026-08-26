"""
DDE-MAE(Wei et al., Interspeech 2025) P1 집계: outputs/ddemae/<run>/best.json (+ logs/<run>.done EXIT=0) → 논문 Table 1/2.
전 run 은 '논문 명세 재구현·가정 런'(코드 없음; 백본 Audio-MAE 만 P0 골든) — 사다리 B2 → S2 → S3 순으로 대조한다.

  .venv/bin/python scripts/reproduction/aggregate_ddemae_results.py
"""
import json
import pathlib
import re
import statistics

# 논문 Table 1/2 (± = 'Confidence Intervals' 캡션, 신뢰수준·run 수 미기재)
PAPER = {
    "icbhi-ddemae-b2":      ("B2  MAE 인코더 + CE 직접 FT (Table 1)",            (64.47, 1.25), (50.72, 0.68), (57.59, 0.91)),
    "icbhi-ddemae-s2":      ("S2  재구성 FT → 분류 (Table 1)",                   (68.89, 1.32), (50.98, 0.76), (59.93, 0.79)),
    "icbhi-ddemae-s3":      ("S3  DDE-MAE 분리 FT → 분류 (Table 1, 주 결과)",    (69.32, 1.27), (53.69, 1.15), (61.50, 0.94)),
    "icbhi-ddemae-nomi":    ("ablation −MI (Table 2)",                           (69.11, 1.20), (53.50, 1.10), (61.30, 0.85)),
    "icbhi-ddemae-nosia":   ("ablation −Siamese (Table 2)",                      (68.54, 1.15), (51.74, 1.02), (60.14, 0.88)),
    "icbhi-ddemae-nomisia": ("ablation −(MI+Siamese) (Table 2 4행, 라벨 중복 → 추정 구성)", (67.61, 1.08), (48.71, 0.98), (58.16, 0.91)),
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
    print("DDE-MAE — 논문 명세 재구현·가정 런 (공개 코드 없음; 백본 Audio-MAE 만 P0 골든). 4 kHz fbank 128(21/128 빈 필터)/1022+2 프레임, 파형 0-패딩(특징값 −1.28, 프레임 57–91% 패딩) 가정.")
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
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
            print(f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f} (std)")
            print(f"  논문 보고 : Sp {psp[0]:6.2f}±{psp[1]:.2f}  Se {pse[0]:6.2f}±{pse[1]:.2f}  Score {psc[0]:6.2f}±{psc[1]:.2f} (CI, run 수 미기재)")
            print(f"  Score 차이: {sc[0]-psc[0]:+.2f}")
    if summary:
        print("\n== 사다리 판정 (B2 → S2 → S3; B2 가 안 맞으면 특징 파이프라인 문제) ==")
        for base in LADDER:
            if base in summary:
                print(f"  {base}: 우리 {summary[base]:.2f} vs 논문 {PAPER[base][3][0]:.2f} ({summary[base]-PAPER[base][3][0]:+.2f})")


if __name__ == "__main__":
    main()
