"""
Lungmix(ICASSP 2025, Ge et al.) P1 집계 — outputs/lungmix/<run>/best.json (+ logs/<run>.done EXIT=0).

**논문 명세 재구현·가정 런** (공식 코드는 증강 함수뿐; epochs/wd/λ2/믹싱 정책/밴드패스/패딩 등 가정 — configs 주석 참고).
**COMB 기준 모델 선택 불가**: 논문 Table I 의 ICBHI/SPR 수치는 COMB(ICBHI+SPR+HF test) 최고 체크포인트이고 HF 는 범위 밖.
툴킷 값은 공식 스플릿 best-on-test(Se 게이트 없음) 이므로 (a) Fig.1 AST SFT(61.35 / 83.67; 선택 기준 미기재 — 단 Fig.1 HF(Test) 열이
Table I 'w/o Mixup' HF 행과 세 값(58.24/62.19/75.34) 동일해 Fig.1 도 Table I 와 같은 run/COMB 선택일 가능성, verdict refuted_claims[2]) 를 baseline 앵커,
(b) Table I 는 '증강 − w/o Mixup' 상대 비교로만 본다. SPRSound 는 7→4 병합 4-class(비표준), 2022 inter-patient test 가정.

행 매핑(저자 질문 #1 미확인): apply_loudness_mask=true(식 3 블렌딩) 런 = Table I 'Lungmix (non-linear)';
apply_loudness_mask=false(공식 함수 출력 = random mask 만) 런 = 'Lungmix (w/o loudness)' (§III-C "using only a random mask").
믹싱 위치: 주 런(*-nonlinear/-none/-woloudness) = mix_stage=raw(원 구간 믹싱 후 9 s; 식 3 maxlen(x_i,x_j) 표기 근거, 가정);
*-padded = 9 s 패딩 후 믹싱 대안(선택 런).

  .venv/bin/python scripts/reproduction/aggregate_lungmix_results.py
"""
import json
import pathlib
import re
import statistics

# run 접두 → (표시 이름, Table I 값 ICBHI→ICBHI / SPR→SPR, 비고)
PAPER_TABLE1 = {
    "icbhi-lungmix-none":      ("AST w/o Mixup",                52.88, None),
    "icbhi-lungmix-nonlinear": ("Lungmix (non-linear)",         53.35, "apply_loudness_mask=true: 식 3 블렌딩, mix_stage=raw (저자 #1/#5 미확인)"),
    "icbhi-lungmix-woloudness": ("Lungmix (w/o loudness)",      58.53, "apply_loudness_mask=false: 공식 함수 출력(random mask 만)"),
    "icbhi-lungmix-padded":    ("Lungmix (non-linear, padded 믹싱)", 53.35, "mix_stage=padded 대안 (9 s 패딩 후 믹싱; 출력 기대 25% 상수 패딩)"),
    "icbhi-lungmix-overlap1":  ("Lungmix (non-linear, overlap=one)", 53.35, "식 8 OR 해석(겹침=1) 대안"),
    "icbhi-lungmix-linear":    ("Lungmix (linear)",             52.13, None),
    "icbhi-lungmix-combined":  ("Lungmix (combined)",           53.78, "λ2 = L_CE/L_mix 가정"),
    "icbhi-lungmix-vanilla":   ("vanilla Mixup",                58.53, "파형 mixup, α=0.5 가정"),
    "icbhi-lungmix-patchmix":  ("Lungmix + Patch-mix",          52.82, None),
    "sprs-lungmix-none":       ("AST w/o Mixup",                80.44, None),
    "sprs-lungmix-nonlinear":  ("Lungmix (non-linear)",         80.61, "apply_loudness_mask=true: 식 3 블렌딩, mix_stage=raw (저자 #1/#5 미확인)"),
    "sprs-lungmix-woloudness": ("Lungmix (w/o loudness)",       76.27, "apply_loudness_mask=false: 공식 함수 출력(random mask 만)"),
    "sprs-lungmix-padded":     ("Lungmix (non-linear, padded 믹싱)", 80.61, "mix_stage=padded 대안"),
    "sprs-lungmix-overlap1":   ("Lungmix (non-linear, overlap=one)", 80.61, "식 8 OR 해석(겹침=1) 대안"),
    "sprs-lungmix-linear":     ("Lungmix (linear)",             77.58, None),
    "sprs-lungmix-combined":   ("Lungmix (combined)",           80.21, None),
    "sprs-lungmix-vanilla":    ("vanilla Mixup",                53.85, "SPR/HF 열 동일값 — 표기 오류 가능"),
    "sprs-lungmix-patchmix":   ("Lungmix + Patch-mix",          79.06, None),
}
FIG1_ANCHOR = {"icbhi": 61.35, "sprs": 83.67}   # AST simple fine-tune (Fig.1; 선택 기준 미기재 — HF 열이 Table I w/o Mixup HF 행과 동일)
TABLE2_ICBHI_NONLINEAR = (79.08, 47.51)          # ICBHI→COMB Sp/Se (범위 밖, 참고)


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def _mean_std(vals):
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main():
    root = pathlib.Path("outputs/lungmix")
    groups = {k: [] for k in PAPER_TABLE1}
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        if run.startswith("smoke"):
            continue
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run); base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups and _completed(root, run):
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
    print("Lungmix P1 집계 — 논문 명세 재구현·가정 런 / 공식 스플릿 best-on-test (COMB 선택 불가) / SPRS 4-class 병합(비표준)")
    for ds in ("icbhi", "sprs"):
        print(f"\n########## {ds.upper()} → {ds.upper()}   (Fig.1 AST SFT 앵커 {FIG1_ANCHOR[ds]:.2f}, 선택 기준 미기재; "
              f"Fig.1 HF 열 = Table I w/o Mixup HF 행 → 같은 COMB 선택 가능성)")
        base_key = f"{ds}-lungmix-none"
        base_rows = groups[base_key]
        base_mean = _mean_std([r[3] for r in base_rows])[0] if base_rows else None
        paper_base = PAPER_TABLE1[base_key][1]
        for key, (name, paper, note) in PAPER_TABLE1.items():
            if not key.startswith(ds):
                continue
            rows = sorted(groups[key])
            if not rows:
                continue
            print(f"\n== {name} ({key}) — {len(rows)} seeds" + (f"   [{note}]" if note else ""))
            for seed, sp, se, sc, ep in rows:
                print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
            sp, se, sc = _mean_std([r[1] for r in rows]), _mean_std([r[2] for r in rows]), _mean_std([r[3] for r in rows])
            line = f"  우리 평균 : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}"
            if key == base_key:
                line += f"   | Fig.1 앵커 {FIG1_ANCHOR[ds]:.2f} 대비 {sc[0]-FIG1_ANCHOR[ds]:+.2f}"
            print(line)
            print(f"  Table I   : {paper:6.2f} (단일값, COMB 기준 best; 직접 비교 불가)")
            if key != base_key and base_mean is not None:
                print(f"  상대 비교 : 우리 (aug − w/o Mixup) {sc[0]-base_mean:+.2f}  vs  논문 {paper-paper_base:+.2f}")
            if key == "icbhi-lungmix-nonlinear":
                print(f"  참고 Table II ICBHI→COMB non-linear Sp/Se {TABLE2_ICBHI_NONLINEAR[0]:.2f}/{TABLE2_ICBHI_NONLINEAR[1]:.2f} (범위 밖)")


if __name__ == "__main__":
    main()
