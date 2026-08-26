"""
LungAdapter P1 집계: outputs/lungadapter/<run>/best.json (+ logs/<run>.done EXIT=0 있으면 완료 여부 확인).

  .venv/bin/python scripts/reproduction/aggregate_lungadapter_results.py

앵커(Interspeech 2024, 5 random runs): Table 1 4-class 80.43±3.11 / 44.37±2.46 / 62.40±1.11 (2.48M/90.01M).
논문 내부 불일치: Figure 2 막대(PDF 벡터 복원, docs/paper_recon_2026-08-26.json verdict) AST+LungAdapter = 75.53±2.83 / 45.21±2.48 / 60.37
→ 두 값을 병기한다. Table 2 ablation: Attn만 77.72/45.05/61.39±1.75, Res만 77.55/44.55/61.05±0.51.
툴킷 내 공정 비교 기준: AST SFT(weighted CE 전체 FT) P1 59.80±0.55 / 논문 59.55 (docs/REPRODUCTION_TARGETS.md).
"""
import json
import pathlib
import re
import statistics

PAPER = {  # run base name → (표시명, Table 값 (Sp,Se,Score,Score std), Fig.2 복원값 or None)
    "icbhi-lungadapter":          ("LungAdapter (Attn+Res) — 4-class 공식 60/40", (80.43, 44.37, 62.40, 1.11), (75.53, 45.21, 60.37)),
    "icbhi-lungadapter-resscale": ("LungAdapter (Attn+Res, res_scale=vector: 파라미터 앵커 2.48M 정확 일치 변형)", (80.43, 44.37, 62.40, 1.11), (75.53, 45.21, 60.37)),
    "icbhi-lungadapter-attnonly": ("ablation: Attention-Adapter 만 (Table 2)",    (77.72, 45.05, 61.39, 1.75), None),
    "icbhi-lungadapter-resonly":  ("ablation: Res-Adapter 만 (Table 2)",          (77.55, 44.55, 61.05, 0.51), None),
}


def _completed(root, run):
    # logs/ 디렉터리가 없으면(런처가 .done 을 안 쓰는 경우) 완료로 간주 — addrsc worker 규약과 동일. 미완료 run 이 섞일 수 있으니 주의.
    done = root / "logs" / f"{run}.done"
    return (not done.parent.exists()) or (done.exists() and "EXIT=0" in done.read_text())


def main():
    root = pathlib.Path("outputs/lungadapter")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        if run.startswith("smoke"):
            continue
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
        name, paper, fig2 = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  논문 Table: Sp {paper[0]:6.2f}  Se {paper[1]:6.2f}  Score {paper[2]:6.2f}±{paper[3]:.2f}   Score 차이 {sc[0]-paper[2]:+.2f}")
            if fig2 is not None:
                print(f"  논문 Fig.2: Sp {fig2[0]:6.2f}  Se {fig2[1]:6.2f}  Score {fig2[2]:6.2f} (PDF 벡터 복원, 내부 불일치)   Score 차이 {sc[0]-fig2[2]:+.2f}")
    print("\n(툴킷 대조 기준: AST SFT weighted-CE 전체 FT P1 59.80±0.55 — scripts/reproduction/aggregate_patchmix_results.py)")


if __name__ == "__main__":
    main()
