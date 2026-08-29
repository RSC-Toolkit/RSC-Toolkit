"""
Prototype(Ren et al., ICASSP 2022) P1 집계: outputs/protonet/<run>/best.json → 논문 Table 3(Test 열) 대비.

논문 프로토콜: 10,000 iteration **최종 iterate** 로 test 평가(별도 best 선택 없음) → `final_*` 를 주 판정으로 쓰고,
툴킷 기본 best-on-test 도 참고로 병기한다. 논문은 단일 run(± 없음).

  .venv/bin/python scripts/reproduction/aggregate_protonet_results.py
"""
import json
import pathlib
import re
import statistics

# 논문 Table 3 Test 열 (SE, SP, AS=Score)
PAPER = {
    "icbhi-protonet-cnn8": ("CNN-8 기준선 (Table 3 1행)", 39.42, 59.72, 49.57),
    "icbhi-protonet":      ("ProtoNet Pt-2D-EleSim-Van w/ BN (논문 'Ours')", 27.78, 72.96, 50.37),
}


def _completed(root, run):
    done = root.parent / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/protonet")
    print("Prototype Learning for Interpretable Respiratory Sound Analysis (ICASSP 2022) — ICBHI 공식 60/40, 4-class")
    print("※ 논문은 단일 run(± 없음). 판정은 **최종 iterate(final_*)** — 공식이 10,000 iteration 모델로만 평가하기 때문.")
    partial = []
    for base, (name, psе, psp, pas) in PAPER.items():
        rows = []
        for d in sorted(root.glob(f"{base}-s[0-9]*")):
            run = d.name
            if not re.fullmatch(rf"{re.escape(base)}-s\d+", run):
                continue
            bj = d / "best.json"
            if not bj.exists():
                continue
            if not _completed(root, run):
                partial.append(run)
                continue
            j = json.loads(bj.read_text())
            rows.append((
                int(run.rsplit("-s", 1)[1]),
                j.get("final_S_e"), j.get("final_S_p"), j.get("final_icbhi_score"),
                j.get("S_e"), j.get("S_p"), j.get("icbhi_score"),
            ))
        rows.sort()
        if not rows:
            continue
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for sd, fse, fsp, fsc, bse, bsp, bsc in rows:
            print(f"  seed {sd}: final Se {fse:6.2f} Sp {fsp:6.2f} Score {fsc:6.2f}  |  best-on-test Score {bsc:6.2f}")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            se, sp, sc = ms(1), ms(2), ms(3)
            bsc = ms(6)
            print(f"  우리 평균(final)      : Se {se[0]:6.2f}±{se[1]:.2f}  Sp {sp[0]:6.2f}±{sp[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            print(f"  우리 평균(best-on-test): Score {bsc[0]:6.2f}±{bsc[1]:.2f} (참고)")
            print(f"  논문 보고             : Se {psе:6.2f}  Sp {psp:6.2f}  Score {pas:6.2f} (단일 run)   Score 차이: {sc[0]-pas:+.2f}")
    if partial:
        print("\n(미완료 run — 집계 제외: " + ", ".join(sorted(partial)) + ")")


if __name__ == "__main__":
    main()
