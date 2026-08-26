"""
GaP-Aug(ICASSP 2024) P1 결과 집계 — **논문 명세 재구현·가정 런** (공식·비공식 코드 없음, 등급 C).

  .venv/bin/python scripts/reproduction/aggregate_gapaug_results.py [--root outputs/gapaug]

읽는 것: outputs/gapaug/<run>/best.json (best-on-test S_p/S_e/icbhi_score/epoch + final_* 최종 iterate) 와 logs/<run>.done 의 EXIT=0.
앵커(공식 60/40, 단일 값·run 수/± 미기재): Table 1 GaP-aug+Mixup(CNN14) Se 58.20 / Sp 77.07 / Score 67.64;
Table 2 Naïve 48.34/64.28/56.31, GaP-aug w/o Mixup 56.49/76.94/66.72, SpecAug w/ Mixup 48.63/79.54/64.09,
PatchMask w/ Mixup 54.88/77.01/65.94, Mixup 55.88/71.82/63.85, SpecAug w/o 50.89/77.96/64.43, PatchMask w/o 54.88/76.18/65.53.
판정(사용자 결정): 5-seed 평균 vs 67.64; 보조 = Naïve 대비 상대 이득(논문 +10.41 w/o mixup, +11.33 w/ mixup).
"""
import argparse
import json
import pathlib
import re
import statistics

# run 접두 → (표시명, 논문 Se, Sp, Score)
PAPER = {
    "icbhi-gapaug":           ("GaP-aug w/ Mixup (Table 1/2)", 58.20, 77.07, 67.64),
    "icbhi-gapaug-nomixup":   ("GaP-aug w/o Mixup",            56.49, 76.94, 66.72),
    "icbhi-gapaug-naive":     ("Naïve (증강 없음)",             48.34, 64.28, 56.31),
    "icbhi-gapaug-specaug":   ("SpecAug w/ Mixup",             48.63, 79.54, 64.09),
    "icbhi-gapaug-patchmask": ("PatchMask w/ Mixup",           54.88, 77.01, 65.94),
    "icbhi-gapaug-mixup":     ("Mixup 만",                     55.88, 71.82, 63.85),
    "icbhi-gapaug-specaug-nomixup":   ("SpecAug w/o Mixup",   50.89, 77.96, 64.43),
    "icbhi-gapaug-patchmask-nomixup": ("PatchMask w/o Mixup", 54.88, 76.18, 65.53),
    # 이탈 플래그 arm: w_trip=0.01(자매 논문 JMIR AI 2025 λ) — 논문 Algorithm 1 은 비가중 합(기본 w_trip=1.0)
    "icbhi-gapaug-wtrip001":  ("GaP-aug w/ Mixup, w_trip=0.01 (이탈 플래그)", 58.20, 77.07, 67.64),
}
NAIVE = "icbhi-gapaug-naive"


def _completed(root: pathlib.Path, run: str) -> bool:
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _ms(vals):
    if not vals:
        return (float("nan"), float("nan"))
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/gapaug")
    ap.add_argument("--include-partial", action="store_true", help=".done 없는 run 도 포함")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        if run.startswith("smoke"):
            continue
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base not in groups:
            continue
        if not _completed(root, run) and not args.include_partial:
            partial.append(run); continue
        d = json.loads(best.read_text())
        groups[base].append(dict(seed=seed, sp=d["S_p"], se=d["S_e"], sc=d["icbhi_score"], ep=d.get("epoch"),
                                 fsp=d.get("final_S_p"), fse=d.get("final_S_e"), fsc=d.get("final_icbhi_score"),
                                 facc=d.get("final_acc@1")))
    print("== GaP-Aug (Chang et al., ICASSP 2024) — 논문 명세 재구현·가정 런 (코드 없음; 가정: min-max 비음수화, Beta α=1, "
          "triplet 가중 1.0@emb(Algo. 1 비가중 합; 0.01 은 이탈 arm), head_input=emb, hann 창, 2회 forward, fade 없음, 앞 절단 등) ==")
    print("   Table 1 보조 앵커 Acc 69.01 은 final_acc@1(최종 iterate) 로만 비교(best 항목엔 acc 없음).")
    print("   프로토콜: 공식 60/40, best-on-test(500 step 마다, Se 게이트 없음) + 최종 iterate(final) 병기. 논문은 단일 값(run 수·± 미기재).")
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
    means = {}
    for base, rows in groups.items():
        if not rows:
            continue
        name, pse, psp, psc = PAPER[base]
        rows.sort(key=lambda r: r["seed"])
        print(f"\n-- {name} ({base}) — {len(rows)} seed --")
        for r in rows:
            fin = f"  | final Sp {r['fsp']:.2f} Se {r['fse']:.2f} Sc {r['fsc']:.2f}" if r["fsc"] is not None else ""
            if r["facc"] is not None:
                fin += f" Acc {r['facc']:.2f}"
            print(f"  seed {r['seed']}: best Sp {r['sp']:6.2f}  Se {r['se']:6.2f}  Score {r['sc']:6.2f} (epoch {r['ep']}){fin}")
        sp, se, sc = _ms([r["sp"] for r in rows]), _ms([r["se"] for r in rows]), _ms([r["sc"] for r in rows])
        means[base] = dict(sp=sp, se=se, sc=sc, n=len(rows))
        print(f"  우리 평균(best) : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
        if all(r["fsc"] is not None for r in rows):
            fsc = _ms([r["fsc"] for r in rows]); fse = _ms([r["fse"] for r in rows]); fsp = _ms([r["fsp"] for r in rows])
            means[base]["fsc"] = fsc
            print(f"  우리 평균(final): Sp {fsp[0]:6.2f}±{fsp[1]:.2f}  Se {fse[0]:6.2f}±{fse[1]:.2f}  Score {fsc[0]:6.2f}±{fsc[1]:.2f}")
        if all(r["facc"] is not None for r in rows):
            facc = _ms([r["facc"] for r in rows])
            print(f"  우리 평균(final Acc): {facc[0]:6.2f}±{facc[1]:.2f}" + ("  (논문 Table 1 Acc 69.01)" if base == "icbhi-gapaug" else ""))
        print(f"  논문 보고       : Sp {psp:6.2f}       Se {pse:6.2f}       Score {psc:6.2f} (단일 값)")
        print(f"  Score 차이(best): {sc[0]-psc:+.2f}")
    if NAIVE in means:
        n_sc = means[NAIVE]["sc"][0]
        print("\n-- Naïve 대비 이득 (보조 판정 기준) --")
        for base, mv in means.items():
            if base == NAIVE:
                continue
            gain = mv["sc"][0] - n_sc
            paper_gain = PAPER[base][3] - PAPER[NAIVE][3]
            print(f"  {PAPER[base][0]:<28}: 우리 {gain:+.2f}  vs 논문 {paper_gain:+.2f}")


if __name__ == "__main__":
    main()
