"""
btscard P1 결과 집계: outputs/btscard/<run>/best.json (+ outputs/btscard/logs/<run>.done EXIT=0) 의 ICBHI IND 지표와
outputs/btscard/<run>/ood.json (eval_btscard_checkpoint.py 산출) 의 SPRSound OOD 지표를 seed 평균±표준편차로 모아
논문 Table 2/3 앵커와 나란히 찍는다.

  .venv/bin/python scripts/reproduction/aggregate_btscard_results.py [--root outputs/btscard]

프로토콜 표기: 툴킷 주 결과 = α=0 고정 · ICBHI 공식 test 기준 best(Score 최고 & Se>5, best-on-test) · seed 1–5 ·
SPRSound OOD = 그 ICBHI-best ckpt 를 2022 inter-patient 4-class 병합 셋에서 평가. 논문 Full 은 α 스윕 중 최고값(Table 3 각주).
"""
import argparse
import json
import pathlib
import re
import statistics

PAPER = {  # (이름, IND Sp/Se/Score, OOD Sp/Se/Score) — 논문 Table 2/3 (5 runs, ± 표기는 분산/표준편차 불명)
    "icbhi-btscard":       ("BTS-CARD Full (코드 레시피 15ep adam)", (84.42, 3.47), (44.83, 2.94), (64.63, 0.57), (82.02, 3.28), (41.90, 4.96), (61.96, 1.50)),
    "icbhi-btscard-paper": ("BTS-CARD Full (논문 레시피 30ep adamw)", (84.42, 3.47), (44.83, 2.94), (64.63, 0.57), (82.02, 3.28), (41.90, 4.96), (61.96, 1.50)),
    "icbhi-btscard-woa":   ("w/o (a) counterfactual", (83.49, None), (43.02, None), (63.25, None), (70.83, None), (46.27, None), (58.55, None)),
    "icbhi-btscard-wob":   ("w/o (b) adversarial", (82.84, None), (45.94, None), (64.33, None), (72.50, None), (46.27, None), (59.39, None)),
    "icbhi-btscard-woc":   ("w/o (c) meta augmentation", (83.55, None), (43.81, None), (63.68, None), (66.99, None), (47.73, None), (56.96, None)),
}
BASELINES = {"BTS (Table 2)": ((81.40, 45.67, 63.54), (67.50, 39.33, 53.42)),
             "Audio-CLAP (Table 2)": ((80.85, 44.67, 62.56), (70.67, 41.90, 56.29))}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _fmt(v):
    return f"{v[0]:6.2f}±{v[1]:.2f}" if v[1] is not None else f"{v[0]:6.2f}      "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/btscard")
    ap.add_argument("--ood_source", default="author", help="ood.json 의 sprsound_<source> 키")
    ap.add_argument("--alpha", default="0.0")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
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
        ood = None
        oj = best.parent / "ood.json"
        if oj.exists():
            o = json.loads(oj.read_text()).get(f"sprsound_{args.ood_source}", {}).get(f"alpha={float(args.alpha):.1f}")
            if o:
                ood = (o["S_p"], o["S_e"], o["icbhi_score"])
        groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch"), ood))
    if partial:
        print("(미완료 run — 집계 제외: " + ", ".join(partial) + ")")
    print(f"프로토콜: α={args.alpha} 고정 · ICBHI 공식 test 기준 best(Se>5, best-on-test) · OOD = ICBHI-best ckpt → SPRSound 2022 inter 4-class ({args.ood_source})")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psp, pse, psc, osp, ose, osc = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)}/5 seeds ==")
        for seed, sp, se, sc, ep, ood in rows:
            o = f"  | OOD Sp {ood[0]:6.2f}  Se {ood[1]:6.2f}  Score {ood[2]:6.2f}" if ood else "  | OOD (ood.json 없음)"
            print(f"  seed {seed}: IND Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep}){o}")
        if len(rows) >= 2:
            def ms(vals):
                return statistics.mean(vals), statistics.stdev(vals)
            sp, se, sc = ms([r[1] for r in rows]), ms([r[2] for r in rows]), ms([r[3] for r in rows])
            print(f"  우리 IND  : Sp {_fmt(sp)}  Se {_fmt(se)}  Score {_fmt(sc)}")
            print(f"  논문 IND  : Sp {_fmt(psp)}  Se {_fmt(pse)}  Score {_fmt(psc)}   Score 차이 {sc[0]-psc[0]:+.2f}")
            oods = [r[5] for r in rows if r[5]]
            if len(oods) >= 2:
                a, b, c = ms([o[0] for o in oods]), ms([o[1] for o in oods]), ms([o[2] for o in oods])
                print(f"  우리 OOD  : Sp {_fmt(a)}  Se {_fmt(b)}  Score {_fmt(c)}  ({len(oods)} seeds)")
                print(f"  논문 OOD  : Sp {_fmt(osp)}  Se {_fmt(ose)}  Score {_fmt(osc)}   Score 차이 {c[0]-osc[0]:+.2f}")
    print("\n앵커(논문 Table 2 베이스라인): " + "; ".join(f"{k} IND {v[0]} / OOD {v[1]}" for k, v in BASELINES.items()))
    print("툴킷 BTS P1(outputs/bts/SUMMARY.txt): 63.91±0.87 (IND)")


if __name__ == "__main__":
    main()
