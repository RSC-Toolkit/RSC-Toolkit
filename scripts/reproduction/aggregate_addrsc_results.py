"""
ADD-RSC P1 집계: outputs/addrsc/<run>/best.json (+ logs/<run>.done EXIT=0).

  .venv/bin/python scripts/reproduction/aggregate_addrsc_results.py

논문(Interspeech 2025, Table 1) AST 변형 85.13/45.94/65.53 은 **단일 seed 0, 코드 스플릿(random.Random(1) 파일 단위 60/40)** 결과로 판단
(논문은 공식 60/40 이라 기재). 주 결과 = 공식 60/40 (툴킷 목적), 비교 = 코드 스플릿 재구성.
"""
import json
import pathlib
import re
import statistics

PAPER = {
    "icbhi-addrsc":      ("ADD-RSC(AST) — 공식 60/40 (주 결과)", None),
    "icbhi-addrsc-code": ("ADD-RSC(AST) — 코드 스플릿(random.Random(1) 파일 단위, 비교용)", (85.13, 45.94, 65.53)),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/addrsc")
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
        name, paper = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            if paper is not None:
                print(f"  논문 보고 : Sp {paper[0]:6.2f}  Se {paper[1]:6.2f}  Score {paper[2]:6.2f} (단일 seed 0)   Score 차이: {sc[0]-paper[2]:+.2f}")
            else:
                print("  논문 수치 없음 (논문 표기는 공식 스플릿이나 코드는 랜덤 파일 스플릿 — 코드 스플릿 결과와 비교)")


if __name__ == "__main__":
    main()
