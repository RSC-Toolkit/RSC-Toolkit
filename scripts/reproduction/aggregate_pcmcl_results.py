"""
PC-MCL P1 결과 집계: outputs/pcmcl/<run>/best.json (+ logs/<run>.done EXIT=0 게이팅) → 논문 Table 1/2/3 앵커 대비.
하이퍼파라미터가 저자 스크립트 부재로 **가정(PAFA 계보)** 이므로 모든 행을 '가정 런' 으로 표기한다.

  .venv/bin/python scripts/reproduction/aggregate_pcmcl_results.py
"""
import json
import pathlib
import re
import statistics

# arXiv 2601.17080 Table 1/2/3 — ICBHI official 60/40 4-class, 5 seeds mean±std (Sp, Se, Score)
PAPER = {
    "icbhi-pcmcl-beats":              ("BEATs + PC-MCL (Table 1/2, 주 결과)",        (79.04, 1.90), (51.71, 2.98), (65.37, 0.73)),
    "icbhi-pcmcl-ast":                ("AST + PC-MCL (Table 1/2)",                   (78.54, 1.87), (46.05, 1.62), (62.30, 0.50)),
    # CE 베이스라인 두 줄은 앵커만 등록 — 대응 config/큐 라인 없음(입력 길이·에폭 미확인, 저자 질문 #7). 실행 전까지 집계에 "미실행" 으로 나타난다.
    "icbhi-pcmcl-beats-ce":           ("BEATs + CE 베이스라인 (Table 2; 설정 미명시, 미실행)", (76.85, 1.88), (48.79, 1.72), (62.82, 0.62)),
    "icbhi-pcmcl-ast-ce":             ("AST + CE 베이스라인 (Table 2; 설정 미명시, 미실행)",   (77.14, 5.43), (41.97, 5.04), (59.55, 0.50)),
    "icbhi-pcmcl-beats-multionly":    ("BEATs ablation Multi only (Table 3)",        (73.10, 2.47), (51.10, 2.97), (62.10, 0.64)),
    "icbhi-pcmcl-beats-concatsingle": ("BEATs ablation Concat only (Table 3)",       (78.05, 2.49), (50.33, 2.93), (64.19, 0.31)),
    "icbhi-pcmcl-beats-nopm":         ("BEATs ablation Concat+Multi, PM 없음 (Table 3)", (76.66, 2.93), (51.91, 2.25), (64.28, 0.62)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/pcmcl")
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
    print("※ 모든 run 은 '가정 런'(학습 스크립트 부재 → PAFA 계보 100ep/bs32/Adam 5e-5/cosine/MA 0.5/nospec 가정) — 저자 확인 전 판정 보류")
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) [가정 런] — {len(rows)}/5 seeds ==")
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
