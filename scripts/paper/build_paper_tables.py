#!/usr/bin/env python3
"""
논문 표 생성기 (제출 대장 B3/B4) — 손으로 옮기지 않는다.

입력: docs/results_master.csv (run 단위 수치) + docs/paper/anchors.csv (표 행 정의·논문값·기재 결정)
      + docs/paper/footnotes.csv (각주 문안) + outputs/<group>/<run>/ood.json (C 표)
출력: docs/paper/tables/table1_ICHBI_Diff.tex (reported | ours | ΔSc) · table2_SPRSound.tex (SPRSound→SPRSound | ICBHI→SPRSound 제로샷, 2026-09-10 통합)
      docs/paper/numbers.md (수치 팩: n·mean±std·Δ·Δ/SE·판정·run 이름·자체 검사)

    python scripts/paper/build_paper_tables.py [--band family|disclosure] [--metrics score|all] [--out docs/paper]

자체 검사: anchors.ledger_score(원장 값) 와 계산값이 0.01 이상 다르면 numbers.md 에 MISMATCH 로 표시하고 종료 코드 1.
"""
from __future__ import annotations
import argparse, csv, json, math, re, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs" / "results_master.csv"
ANCHORS = ROOT / "docs" / "paper" / "anchors.csv"
FOOTNOTES = ROOT / "docs" / "paper" / "footnotes.csv"
META = ROOT / "docs" / "paper" / "model_meta.csv"      # Patch-Mix 스타일 표용 메타 (architecture / pretrain / venue / cite)
PARAMS = ROOT / "docs" / "paper" / "params.csv"        # 표 2 #Param (M) 열 원천 — scripts/paper/count_params.py 가 생성 (2026-09-15)
BAND_ORDER = ["Fine-tuning", "Augmentation", "Contrastive / auxiliary loss", "Foundation / PEFT", "Architecture / protocol"]
BAND_SHORT = {"Fine-tuning": "Fine-tuning", "Augmentation": "Aug.", "Contrastive / auxiliary loss": "Contrastive",
              "Foundation / PEFT": "FM / PEFT", "Architecture / protocol": "Arch."}   # 회전 라벨 (2026-09-08 사용자: Team 밴드·표기 없음)

DISCLOSURE_LABEL = {"full": "Code + training scripts released", "code": "Code released",
                    "partial": "Partial release / paper-code mismatch", "none": "No code released"}
DISCLOSURE_ORDER = ["full", "code", "partial", "none"]
MARK = {"team": r"$^{\dagger}$", "assume": r"$^{\ast}$", "variant": r"$^{\ddagger}$"}
MARK_SYM = {"team": r"\dagger", "assume": r"\ast", "variant": r"\ddagger"}


def marks_tex(flags: str) -> str:
    """flags 'team;assume' → $^{\dagger\ast}$ (한 수식 그룹으로 합쳐 $$ 인접을 피한다)"""
    syms = [MARK_SYM[f] for f in flags.split(";") if f]
    return ("$^{" + "".join(syms) + "}$") if syms else ""


def num2(x):
    """논문 보고값 문자열을 소수 2자리로 통일 (비수치·빈 값은 그대로/--)"""
    if x in ("", None):
        return "--"
    try:
        return f"{float(x):.2f}"
    except ValueError:
        return x


def family_of(run_name: str) -> str:
    return re.sub(r"-s\d+$", "", run_name)


def load_master():
    rows = list(csv.DictReader(open(MASTER)))
    fam = {}
    for r in rows:
        fam.setdefault(family_of(r["run_name"]), []).append(r)
    return rows, fam


def fnum(x):
    return float(x) if x not in ("", None) else None


def ood_value(run_dir: str, path: str, key: str):
    d = json.load(open(ROOT / run_dir / "ood.json"))
    for seg in path.split("/"):
        d = d[seg]
    return float(d[key])


def aggregate(anchor, fam):
    rows = fam.get(anchor["family"], [])
    src = anchor["metric_source"]
    vals = {"sp": [], "se": [], "score": []}
    runs = []
    for r in rows:
        if src == "best":
            sp, se, sc = fnum(r["s_p"]), fnum(r["s_e"]), fnum(r["icbhi_score"])
        elif src == "final":
            sp, se, sc = fnum(r["final_S_p"]), fnum(r["final_S_e"]), fnum(r["final_icbhi_score"])
        elif src.startswith("ood:"):
            p = src[4:]
            if not (ROOT / r["run_dir"] / "ood.json").exists():
                continue  # OOD 미평가 run 은 집계에서 제외 (n 에 반영)
            sp, se, sc = (ood_value(r["run_dir"], p, k) for k in ("S_p", "S_e", "icbhi_score"))
        else:
            raise ValueError(src)
        if sc is None:
            continue
        vals["sp"].append(sp); vals["se"].append(se); vals["score"].append(sc)
        runs.append((r["run_name"], r["done"]))
    n = len(vals["score"])
    def ms(v):
        return (st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)) if v else (None, None)
    out = {k: ms(v) for k, v in vals.items()}
    out["n"] = n
    out["runs"] = runs
    out["done_all"] = all(d in ("1", "done") for _, d in runs)
    # 부가 지표: AMS-D 논문 산식(paper_score), RespireNet/Prototype 대조값
    extra = {}
    if anchor["family"].endswith("-amsd") and src == "best":
        ps = [fnum(r["paper_score"]) for r in rows if r.get("paper_score")]
        pse = [fnum(r["paper_S_e"]) for r in rows if r.get("paper_S_e")]
        if ps:
            extra["binary_score"] = ms(ps); extra["binary_se"] = ms(pse)
    if src == "final":
        bs = [fnum(r["icbhi_score"]) for r in rows if r.get("icbhi_score")]
        if bs:
            extra["best_on_test"] = ms(bs)
    out["extra"] = extra
    return out


def fmt(m, s=None, bold=False, tex=True):
    if m is None:
        return "--" if tex else "—"
    if s is None:
        return f"{m:.2f}"
    core = f"{m:.2f}"
    core = (r"\textbf{" + core + "}") if (bold and tex) else (f"**{core}**" if bold and not tex else core)
    pm = r"$\pm$" if tex else "±"
    return f"{core} {pm} {s:.2f}" if tex else f"{core}{pm}{s:.2f}"


def tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
             .replace("±", r"$\pm$").replace("→", r"$\to$").replace("≥", r"$\ge$"))


def delta_str(d, tex=True):
    if d is None:
        return "--" if tex else "—"
    sign = "+" if d >= 0 else ("$-$" if tex else "−")
    return f"{sign}{abs(d):.2f}"


def paper_cell(a, metrics, tex=True):
    sp, se, sc, sd = (a["paper_sp"], a["paper_se"], a["paper_score"], a["paper_std"])
    if not sc:
        return "--" if tex else "—"
    score = num2(sc) + ((r" $\pm$ " if tex else "±") + num2(sd) if sd else "")
    if metrics == "score":
        return score
    if sp and se:
        return f"{num2(sp)} / {num2(se)} / {score}"
    return ("-- / -- / " if tex else "— / — / ") + score


def ours_cell(agg, metrics, tex=True):
    if agg["n"] == 0:
        return "--" if tex else "—"
    sc = fmt(*agg["score"], bold=True, tex=tex)
    if metrics == "score":
        return sc
    return f"{fmt(*agg['sp'], tex=tex)} / {fmt(*agg['se'], tex=tex)} / {sc}"


def label_cell(a, tex=True):
    marks = marks_tex(a["flags"]) if tex else \
            "".join({"team": "†", "assume": "*", "variant": "‡"}[f] for f in a["flags"].split(";") if f)
    fn = ("".join(r"\textsuperscript{" + x + "}" for x in a["fn"].split(";") if x) if tex
          else "".join(f"^{x}" for x in a["fn"].split(";") if x))
    name = tex_escape(a["model"]) if tex else a["model"]
    var = a["variant"]
    if tex:
        return name + marks + fn + ((r" {\scriptsize " + tex_escape(var) + "}") if var else "")
    return name + marks + fn + (f" ({var})" if var else "")


def build_table(anchors, fam, table, band_mode, metrics, footnotes):
    rows = [a for a in anchors if a["table"] == table]
    if band_mode == "disclosure":
        rows.sort(key=lambda a: (DISCLOSURE_ORDER.index(a["disclosure"]), int(a["order"])))
        band_of = lambda a: DISCLOSURE_LABEL[a["disclosure"]]
    else:
        rows.sort(key=lambda a: int(a["order"]))
        band_of = lambda a: a["band"]
    ncol = 4 if table != "B" else 4
    head = {"A": ("Method", "Paper", "Ours (5 seeds)", r"$\Delta$"),
            "B": ("Method", "Paper", "Ours (5 seeds)", r"$\Delta$"),
            "C": ("Method", "Paper", "Ours (5 seeds)", r"$\Delta$")}[table]
    unit = "Score" if metrics == "score" else "Sp / Se / Score"
    caption = {"A": f"ICBHI 2017, official 60/40 split, 4-class ({unit}). Best-on-test as in each paper; Prototype uses the final iterate.",
               "B": f"SPRSound 2022 inter-patient test, 7-class ({unit}). Paper column only where the paper reports SPRSound.",
               "C": f"ICBHI-trained checkpoints evaluated zero-shot on SPRSound inter-patient test ({unit})."}[table]
    L = [r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{" + caption + "}", r"\label{tab:" + {"A": "icbhi", "B": "sprsound", "C": "ood"}[table] + "}",
         r"\begin{tabular}{@{}l" + "r" * (ncol - 1) + "@{}}", r"\toprule",
         " & ".join(head) + r" \\", r"\midrule"]
    cur = None
    used_fn = set()
    aggs = {}
    for a in rows:
        b = band_of(a)
        if b != cur:
            L.append(r"\multicolumn{" + str(ncol) + r"}{@{}l}{\textit{" + tex_escape(b) + r"}} \\")
            cur = b
        agg = aggregate(a, fam); aggs[a["family"] + "|" + a["metric_source"]] = agg
        d = None
        if a["delta"] == "score" and a["paper_score"] and agg["n"]:
            d = agg["score"][0] - float(a["paper_score"])
        for x in a["fn"].split(";"):
            if x: used_fn.add(x)
        for f in a["flags"].split(";"):
            if f: used_fn.add(f)
        L.append(" & ".join([label_cell(a), paper_cell(a, metrics), ours_cell(agg, metrics), delta_str(d)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    notes = [footnotes[k] for k in ["team", "assume", "variant"] if k in used_fn]
    notes += [r"\textsuperscript{" + k + "} " + footnotes[k] for k in sorted((x for x in used_fn if x.isdigit()), key=int)]
    if notes:
        L += [r"\par\smallskip\begin{minipage}{\textwidth}\scriptsize", (r" ".join(
            (MARK.get(k, "") + " " + footnotes[k]) for k in ["team", "assume", "variant"] if k in used_fn)),
              " ".join(r"\textsuperscript{" + k + "} " + footnotes[k] for k in sorted((x for x in used_fn if x.isdigit()), key=int)),
              r"\end{minipage}"]
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n", aggs


def load_meta():
    return {r["family"]: r for r in csv.DictReader(open(META))}


def _rank_marks(vals):
    """vals: list of float|None (비교 불가 행은 None). 최고→bold, 차고→underline."""
    order = sorted((i for i, v in enumerate(vals) if v is not None), key=lambda i: -vals[i])
    return {**({order[0]: "best"} if order else {}), **({order[1]: "second"} if len(order) > 1 else {})}


def _mark(core: str, kind):
    return {"best": r"\textbf{" + core + "}", "second": r"\underline{" + core + "}"}.get(kind, core)


def _ours_pm(m, s, kind=None, pm=True):
    if m is None:
        return "--"
    core = _mark(f"{m:.2f}", kind)
    # 2026-09-15 사용자 요청: 표준편차는 아래첨자로 작게 (mean${}_{\pm sd}$). 09-14 의 본문 크기 표기를 대체.
    return core + (r"${}_{\pm" + f"{s:.2f}" + "}$" if pm and s is not None else "")


DELTA_COLORS = r"\definecolor{dpos}{RGB}{0,110,60}\definecolor{dneg}{RGB}{190,30,30}"  # 표 파일 상단에 삽입 (xcolor 필요)


def delta_color_tex(d):
    """Δ(우리 − 논문) 을 색으로: ≥0 초록(dpos), <0 빨강(dneg). None → --"""
    if d is None:
        return "--"
    return (r"\textcolor{dpos}{" if d >= 0 else r"\textcolor{dneg}{") + delta_str(d) + "}"


def deltas_of(a, agg):
    """세 지표 Δ. 같은 프로토콜(anchors.delta == 'score')이고 논문값이 있는 지표만."""
    if a["delta"] != "score" or not agg["n"]:
        return (None, None, None)
    out = []
    for k, col in (("sp", "paper_sp"), ("se", "paper_se"), ("score", "paper_score")):
        out.append((agg[k][0] - float(a[col])) if a[col] else None)
    return tuple(out)


PM_TABLES = {  # table → (file stem, caption 앞부분, label)
    "A": ("table1_ICHBI_Diff", "ICBHI 2017 (official 60/40 split, 4-class cycle classification)", "icbhi_diff"),
}   # B(SPRSound→SPRSound)·C(ICBHI→SPRSound) 는 2026-09-10 부터 통합 표 MERGED(table2_SPRSound) 한 장으로 — build_merged


def _rep(x):
    """논문 보고값 셀: 없으면 빈칸(사용자 지시), 있으면 소수 2자리"""
    return "" if x in ("", None) else num2(x)


def build_patchmix(anchors, fam, table, meta, footnotes):
    """사용자 형식(2026-09-08): method · architecture · pretrain · venue | reported Sp/Se/Sc | ours Sp/Se/Sc (5-seed mean±std) | ΔSc(색).
    행 = anchors.pm_order 가 있는 행, 그 순서. 논문값이 없는 행은 reported·ΔSc 를 비운다. 같은 프로토콜이 아니면 ΔSc = --.
    best/second 는 comparable=1 행끼리, 우리 값 기준. 반환 aggs 는 표의 전 행(numbers.md 용)."""
    all_rows = [a for a in anchors if a["table"] == table]
    aggs = {a["family"] + "|" + a["metric_source"]: aggregate(a, fam) for a in all_rows}
    rows = sorted((a for a in all_rows if a.get("pm_order")), key=lambda a: (BAND_ORDER.index(a["band"]), float(a["pm_order"])))
    def agg_of(a): return aggs[a["family"] + "|" + a["metric_source"]]
    def comp(a): return meta[a["family"]]["comparable"] == "1"
    marks = {k: _rank_marks([agg_of(a)[k][0] if (comp(a) and agg_of(a)["n"]) else None for a in rows])
             for k in ("sp", "se", "score")}
    stem, dsdesc, label = PM_TABLES[table]
    colspec = "p{2pt}llcl" + "ccc" + "ccc" + "c"
    head = [r"& \multirow{2}{*}{method} & \multirow{2}{*}{architecture} & \multirow{2}{*}{pretrain} & \multirow{2}{*}{venue} & "
            r"\multicolumn{3}{c}{reported in paper} & \multicolumn{3}{c}{ours (5 seeds)} & \multirow{2}{*}{$\Delta S_c$} \\",
            r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
            r"& & & & & $S_p$ & $S_e$ & $S_c$ & $S_p$ & $S_e$ & $\mathbf{S_c}$ & \\"]
    common = (r" ``Reported'' copies the numbers from each paper (single run or mean over the paper's seeds; blank where the paper reports "
              r"nothing for this setting); ``ours'' is the toolkit re-implementation with five seeds (mean\,$\pm$\,std). "
              r"$\Delta S_c$ = ours $-$ reported Score (\textcolor{dpos}{green}: at or above, \textcolor{dneg}{red}: below), shown only where the "
              r"two are measured on the same split with the same metric. \textbf{Bold} and \underline{underline} mark the highest and second-highest "
              r"``ours'' value per column among directly comparable rows.")
    common = common + r" Rows are grouped by the type of contribution (rotated labels)."
    cap = {"A": CAPTION_A,
           "B": "Reported vs.\\ reproduced results on " + dsdesc + ".",
           "C": "Reported vs.\\ reproduced " + dsdesc + "."}[table] + ("" if table == "A" else common)
    L = [DELTA_COLORS, r"\begin{table*}[!t]", r"\centering", r"\caption{" + cap + "}", r"\label{tab:" + label + "}",
         r"\renewcommand{\arraystretch}{1}\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{", r"\begin{tabular}{" + colspec + "}", r"\toprule", *head, r"\hline \midrule"]
    used_fn = set()
    band_sizes = {b: sum(1 for a in rows if a["band"] == b) for b in BAND_ORDER}
    cur_band = None
    for i, a in enumerate(rows):
        agg = agg_of(a); m = meta[a["family"]]
        fns = [x for x in a["fn"].split(";") if x]
        if table == "A":
            fns = sorted(_local(fns, "A"), key=int)
        used_fn.update(fns)
        fn = (r"\textsuperscript{" + ",".join(fns) + "}") if fns else ""
        name = tex_escape(m["method"]) + r"~\cite{" + m["cite"].replace(" ", "") + "}" + fn
        if a["band"] != cur_band:
            if cur_band is not None:
                L.append(r"\midrule")
            cur_band = a["band"]
            first = (r"\multirow{" + str(band_sizes[cur_band]) + r"}{*}{\rotatebox[origin=c]{90}{\scriptsize\textbf{" + BAND_SHORT[cur_band] + "}}}")
        else:
            first = ""
        cells = [first, name, m["arch"], tex_escape(m["pretrain"]).replace("+", r"\,+\,"), r"\textit{" + m["venue"] + "}",
                 _rep(a["paper_sp"]), _rep(a["paper_se"]), _rep(a["paper_score"]),
                 _ours_pm(*agg["sp"], marks["sp"].get(i)), _ours_pm(*agg["se"], marks["se"].get(i)),
                 _ours_pm(*agg["score"], marks["score"].get(i))]
        if not a["paper_score"]:
            cells.append("")
        elif a["delta"] != "score" or not agg["n"]:
            cells.append("--")
        else:
            cells.append(delta_color_tex(agg["score"][0] - float(a["paper_score"])))
        if m.get("backbone_cite"):   # 2026-09-15 사용자: backbone 논문은 표에서 인용하지 않고(본문에서) 행 위 주석으로만 남긴다
            L.append("% backbone: " + m["backbone_cite"].replace(" ", ""))
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    key = (lambda k: FN_KEY["A"] + k) if table == "A" else (lambda k: k)
    notes = [r"\textsuperscript{" + k + "} " + footnotes[key(k)] for k in sorted((x for x in used_fn if x.isdigit()), key=int)]
    if table == "A":
        notes.append(ABBREV_LINE)   # 번호 없는 공통 범례 — 표 1 아래 한 번만 (표 2 캡션은 "Abbreviations as in Table 1")
    if notes:
        L += [r"\par\vspace{2pt}\begin{minipage}{\linewidth}\scriptsize", " ".join(notes), r"\end{minipage}"]
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n", aggs

# ── 통합 표 2 (2026-09-10 사용자 지시): 표 2(SPRSound→SPRSound) 와 표 4(ICBHI→SPRSound 제로샷) 를 한 표로 합치고 논문 보고값 열은 없앤다 ──
MERGED = ("table2_SPRSound", "sprsound")     # (파일 stem, \label)
MERGE_KEY_OVERRIDES = {"icbhi-btscard-paper": "btscard", "icbhi-lungsrad-dualpaper": "lungsrad-dual",
                       "icbhi-mvst-official-fusion": "mvst-fusion"}   # family 접두사(sprs-/icbhi-)만 떼서는 짝이 안 맞는 것
MERGE_ORDER_HINT = {"m2d-base": 13.5}        # 표 1(A) 에 짝이 없는 행의 자리 — icbhi-m2dx-s5(A pm_order 14) 바로 앞
METHOD_LEVEL_FN = {"1", "2", "12", "13"}     # 두 열 모두에 걸리는 각주(코드 없음·하이퍼파라미터·정의 불가) → 방법 이름에; 나머지는 해당 열의 Sc 셀에
CAPTION_FN = {"8"}                           # OOD 프로토콜(7→4 병합·α=0) 은 캡션 문장으로 옮겼다
# 2026-09-14 사용자 지시 — 표마다 각주를 다시 번호 매긴다. anchors.fn 의 전역 id → 표 로컬 번호; 맵에 없는 전역 id 는 캡션/본문으로 옮겨져 표식이 없다.
#   표 1: 1·2(코드 부재/부분 재구성/추정 설정)→1, 3(파일 단위 분할)→2, 5(AMS-D 이진 지표)→3, 6(Prototype final iterate)→4; 4(CLAP 사전계산 피처)는 본문으로.
#   표 2: 7(BTS 4필드)→1, 13(PC-MCL 6속성)→2, 14(QLung log4)→3, 6+10(Prototype final/best)→4, 11(M2D 마지막 배치)→5; 1·2·9·12 는 캡션으로.
#   2026-09-14 추가: 15(OPERA-CT 전체 미세조정 = 툴킷 정의 레시피)→표 1 로컬 5·표 2 로컬 6, 16(M2D-X ICBHI 특화 ckpt 의 SPRSound 미세조정)→표 2 로컬 7.
#   2026-09-14 저녁(사용자): 표 1 각주 1(코드 부재/재구성; 전역 1·2) 제거 → 본문. Prototype·LungMix 표에서 제거(전역 6·10·9 미사용).
#   n/a 사유는 캡션 대신 각주로: 12(SG-SCL 단일 청진기) → t2.7, 17(RespireNet 장치별 설계·SPRSound 로더 없음) → t2.8.
FN_MAP = {"A": {"3": "1", "5": "2"},
          "M": {"7": "1", "13": "2", "14": "3", "11": "4", "12": "5", "17": "6"}}
# 2026-09-15 사용자: M2D-X SPRSound 미세조정 각주(전역 16 = 표 2 로컬 5)도 삭제 → 표 2 각주 1~6.
# 2026-09-15 사용자: OPERA-CT 전체 미세조정 각주(전역 15 = 표 1 로컬 3 · 표 2 로컬 5)는 의미 없음 → 두 표에서 제거, 뒤 번호 당김(표 2 6·7·8 → 5·6·7).
FN_KEY = {"A": "t1.", "M": "t2."}   # footnotes.csv 의 로컬 문안 id 접두사
RANK_EXCLUDE = {"C": {"11"}}          # 각주 5(M2D 마지막 배치 탈락, 1,408/1,429) 인 OOD 행은 전체 테스트셋 순위(굵게/밑줄)에서 제외 (2026-09-14)
# 2026-09-14 사용자 제공 캡션 문안(그대로 사용)
CAPTION_A = (r"Reported and toolkit results on ICBHI 2017. Toolkit results use the official 60/40 split and four-class scoring (five seeds; "
             r"mean\,$\pm$\,SD). $\Delta S_c$ = toolkit $-$ reported Score, shown only for comparable splits, metrics, and checkpoint selection. "
             r"Bold and underlined values mark the highest and second-highest comparable toolkit means per column.")
CAPTION_M = (r"Toolkit results on the SPRSound 2022 inter-patient test set (1,429 events unless noted; five seeds; mean\,$\pm$\,SD). "
             r"SPRSound $\to$ SPRSound uses seven classes. ICBHI $\to$ SPRSound is zero-shot, using four merged classes and "
             r"checkpoints selected only on ICBHI test data. Bold and underlined values mark the two highest means per column among results with "
             r"matching label spaces and test samples. n/a: not available (see notes); \mbox{--}: not run. "
             r"Abbreviations as in Table~\ref{tab:icbhi_diff}.")
# 2026-09-14 사용자 축약안. 표 1 아래 번호 없는 범례 한 줄(ABBREV_LINE); 자세한 정의는 본문(docs/paper/body_text_moved_from_footnotes.md).
ABBREV_LINE = (r"\textit{Abbreviations:} IN, ImageNet; AS, AudioSet; LA, LAION-Audio-630K; Resp, respiratory audio; FM, foundation model; "
               r"PEFT, parameter-efficient fine-tuning; IS, Interspeech. \textit{Row groups:} Fine-tuning, plain fine-tuning of a pre-trained backbone; "
               r"Aug., data augmentation; Contrastive, contrastive or auxiliary training objectives; FM / PEFT, foundation models and parameter-efficient "
               r"fine-tuning; Arch., architecture or training protocol.")


def _local(ids, table):
    return {FN_MAP[table][x] for x in ids if x in FN_MAP[table]}


def merge_key(family: str) -> str:
    return MERGE_KEY_OVERRIDES.get(family, re.sub(r"^(sprs|icbhi)-", "", family))


def load_params():
    """docs/paper/params.csv → {family: params_total(int)}. 파일이 없으면 빈 dict(열은 -- 로 채워진다)."""
    if not PARAMS.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(PARAMS)):
        if r.get("params_total"):
            out[r["family"]] = int(r["params_total"])
    return out


def _param_value(b, c, params):
    """정의(2026-09-15 사용자 결정) = 체크포인트의 전체 파라미터(동결 백본·대조 projection head 포함); 캡션에 정의 문장은 넣지 않는다.
    통합 행의 #Param: SPRSound 쪽(B) 모델, 없으면 ICBHI 쪽(C). 두 쪽은 분류 헤드(7 vs 4 class)만 달라 0.1 M 해상도에서 같다(count_params 가 기록)."""
    for a in (b, c):
        if a is not None and a["family"] in params:
            return params[a["family"]] / 1e6
    return None


def _param_cell(b, c, params):
    v = _param_value(b, c, params)
    return "--" if v is None else f"{v:.1f}"


def merged_rows(anchors):
    """B(SPR→SPR)·C(ICBHI→SPR) 행을 merge_key 로 짝지어 [(key, B행|None, C행|None)] 을 표 1 순서(밴드 → A pm_order)로 돌려준다."""
    by = {t: {merge_key(a["family"]): a for a in anchors if a["table"] == t and a.get("pm_order")} for t in "ABC"}
    out = []
    for k in set(by["B"]) | set(by["C"]):
        b, c = by["B"].get(k), by["C"].get(k)
        ref = by["A"].get(k) or c or b
        if k in by["A"]:
            pos = float(by["A"][k]["pm_order"])
        elif k in MERGE_ORDER_HINT:
            pos = MERGE_ORDER_HINT[k]
        else:
            raise KeyError(f"merged table: no row order for '{k}' (add it to MERGE_ORDER_HINT)")
        out.append((BAND_ORDER.index(ref["band"]), pos, k, b, c))
    out.sort(key=lambda t: t[:3])
    return [(k, b, c) for _, _, k, b, c in out]


def _fns(a):
    return {x for x in (a["fn"].split(";") if a else []) if x} - CAPTION_FN


def _split_fns(b, c):
    """표 2 각주는 전부 해당 열의 Sc 셀에(로컬 번호 FN_MAP['M']). 맵에 없는 전역 각주(1·2·9·12)는 캡션으로 옮겨져 표식이 없다.
    반환: (이름 각주 = 항상 빈 집합, {"B": …, "C": …})"""
    return set(), {"B": _local(_fns(b), "M"), "C": _local(_fns(c), "M")}


def _fallback_lookup(anchors):
    """(표, family) → 그 표의 anchor 행. na_fallback 이 가리키는 family 의 값을 빠진 쪽 칸에 채울 때 쓴다."""
    return {(a["table"], a["family"]): a for a in anchors if a["table"] in "BC" and a.get("pm_order")}


def _sup(fns, html=False):
    s = ",".join(sorted(fns, key=int))
    return "" if not s else (f"<sup>{s}</sup>" if html else r"\textsuperscript{" + s + "}")


def build_merged(anchors, fam, meta, footnotes):
    """통합 표 2: method · architecture · pretrain · venue | SPRSound→SPRSound ours Sp/Se/Sc | ICBHI→SPRSound(zero-shot) ours Sp/Se/Sc.
    한쪽이 없는 행: na_fn 이 있으면 n/a + 각주(정의 불가·미학습 사유), 없으면 --(이번 연구에서 안 돌림). best/second 는 열별로 comparable=1 행끼리.
    반환 (tex, aggs_B, aggs_C) — aggs 는 numbers.md 용 전 행."""
    aggs = {t: {a["family"] + "|" + a["metric_source"]: aggregate(a, fam) for a in anchors if a["table"] == t} for t in "BC"}
    rows = merged_rows(anchors)
    def agg_of(a, t): return aggs[t][a["family"] + "|" + a["metric_source"]]
    def comp(a): return meta[a["family"]]["comparable"] == "1"
    marks = {}
    for t, i in (("B", 1), ("C", 2)):
        for k in ("sp", "se", "score"):
            marks[t, k] = _rank_marks([agg_of(r[i], t)[k][0] if (r[i] is not None and comp(r[i]) and agg_of(r[i], t)["n"]
                                                                and not (_fns(r[i]) & RANK_EXCLUDE.get(t, set()))) else None
                                       for r in rows])
    fallback = _fallback_lookup(anchors)
    params = load_params()
    stem, label = MERGED
    colspec = "p{2pt}llr" + "ccc" + "ccc"   # 2026-09-14: 표 1 과 중복되는 pretrain·venue 열 제거 · 2026-09-15: #Param (M) 열(사용자 요청)
    head = [r"& \multirow{2}{*}{method} & \multirow{2}{*}{architecture} & \multirow{2}{*}{\#Param (M)} & "
            r"\multicolumn{3}{c}{SPRSound $\to$ SPRSound} & \multicolumn{3}{c}{ICBHI $\to$ SPRSound (zero-shot)} \\",
            r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
            r"& & & & $S_p$ & $S_e$ & $\mathbf{S_c}$ & $S_p$ & $S_e$ & $\mathbf{S_c}$ \\"]
    cap = CAPTION_M
    L = [r"\begin{table*}[!t]", r"\centering", r"\caption{" + cap + "}", r"\label{tab:" + label + "}",
         r"\renewcommand{\arraystretch}{1}\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{", r"\begin{tabular}{" + colspec + "}", r"\toprule", *head, r"\hline \midrule"]
    used_fn = set()
    band_of = lambda r: (r[2] or r[1])["band"]
    bands = [meta_band for meta_band in BAND_ORDER]
    # 밴드는 표 1 과 같은 기준(A 행의 band) — merged_rows 가 그 순서로 정렬했으므로 여기서는 A 기준 band 를 다시 구한다
    A = {merge_key(a["family"]): a for a in anchors if a["table"] == "A" and a.get("pm_order")}
    row_band = [A[k]["band"] if k in A else band_of((k, b, c)) for k, b, c in rows]
    band_sizes = {bd: row_band.count(bd) for bd in bands}
    cur_band = None
    for i, (k, b, c) in enumerate(rows):
        ref = c or b
        m = meta[ref["family"]]
        name_fn, side_fn = _split_fns(b, c)
        used_fn |= name_fn | side_fn["B"] | side_fn["C"]
        name = tex_escape(m["method"]) + r"~\cite{" + m["cite"].replace(" ", "") + "}" + _sup(name_fn)
        if row_band[i] != cur_band:
            if cur_band is not None:
                L.append(r"\midrule")
            cur_band = row_band[i]
            first = (r"\multirow{" + str(band_sizes[cur_band]) + r"}{*}{\rotatebox[origin=c]{90}{\scriptsize\textbf{" + BAND_SHORT[cur_band] + "}}}")
        else:
            first = ""
        cells = [first, name, m["arch"], _param_cell(b, c, params)]
        for t, a, other in (("B", b, c), ("C", c, b)):
            if a is None:
                na = (other.get("na_fn") or "") if other else ""   # 사유는 각주(FN_MAP M)로; 캡션은 "n/a: not available" 만
                fb = fallback.get((t, (other.get("na_fallback") or "") if other else ""))
                if fb is not None:  # 빠진 쪽을 다른 family 값으로 채운다; 순위 표시에서는 제외
                    agg = aggregate(fb, fam)
                    cells += [_ours_pm(*agg["sp"]), _ours_pm(*agg["se"]), _ours_pm(*agg["score"])]
                    continue
                loc = _local({na}, "M") if na else set(); used_fn |= loc
                cells.append(r"\multicolumn{3}{c}{" + ("n/a" if na else "--") + _sup(loc) + "}")
                continue
            agg = agg_of(a, t)
            if not agg["n"]:                                    # 행은 있으나 run 이 없다(RespireNet·Gap-Aug: 표 1 과 목록을 맞추기 위한 자리) → n/a(각주) 또는 --
                na = a.get("na_fn") or ""
                loc = _local({na}, "M") if na else set(); used_fn |= loc
                cells.append(r"\multicolumn{3}{c}{" + ("n/a" if na else "--") + _sup(loc) + "}")
                continue
            cells += [_ours_pm(*agg["sp"], marks[t, "sp"].get(i)), _ours_pm(*agg["se"], marks[t, "se"].get(i)),
                      _ours_pm(*agg["score"], marks[t, "score"].get(i)) + _sup(side_fn[t])]
        if m.get("backbone_cite"):   # 2026-09-15 사용자: backbone 논문은 표에서 인용하지 않고(본문에서) 행 위 주석으로만 남긴다
            L.append("% backbone: " + m["backbone_cite"].replace(" ", ""))
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    notes = [r"\textsuperscript{" + k + "} " + footnotes[FN_KEY["M"] + k] for k in sorted((x for x in used_fn if x.isdigit()), key=int)]
    if notes:
        L += [r"\par\vspace{2pt}\begin{minipage}{\linewidth}\scriptsize", " ".join(notes), r"\end{minipage}"]
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n", aggs["B"], aggs["C"]


def merged_html(anchors, fam, meta, footnotes, tex2html):
    """통합 표 2 의 HTML 미리보기(종합 대장) — build_merged 와 같은 행·수치·표시."""
    aggs = {t: {a["family"] + "|" + a["metric_source"]: aggregate(a, fam) for a in anchors if a["table"] == t} for t in "BC"}
    rows = merged_rows(anchors)
    def agg_of(a, t): return aggs[t][a["family"] + "|" + a["metric_source"]]
    def comp(a): return meta[a["family"]]["comparable"] == "1"
    marks = {}
    for t, i in (("B", 1), ("C", 2)):
        for k in ("sp", "se", "score"):
            marks[t, k] = _rank_marks([agg_of(r[i], t)[k][0] if (r[i] is not None and comp(r[i]) and agg_of(r[i], t)["n"]
                                                                and not (_fns(r[i]) & RANK_EXCLUDE.get(t, set()))) else None
                                       for r in rows])
    A = {merge_key(a["family"]): a for a in anchors if a["table"] == "A" and a.get("pm_order")}
    row_band = [A[k]["band"] if k in A else (c or b)["band"] for k, b, c in rows]
    fallback = _fallback_lookup(anchors)
    params = load_params()
    used = set()
    L = ['<div class="pm pm-M">', f'<p class="pm-title"><b>table2_SPRSound · SPRSound→SPRSound (7-class) | ICBHI→SPRSound 제로샷 (7→4 병합, 1,429 event) · reported 열 없음</b> · {len(rows)} 행</p>',
         '<div class="tbl"><table class="pmt"><thead>',
         '<tr class="grp"><th colspan="4"></th><th colspan="3">SPRSound → SPRSound (ours, 5 seeds)</th><th colspan="3">ICBHI → SPRSound zero-shot (ours, 5 seeds)</th></tr>',
         "<tr><th></th><th>method</th><th>architecture</th><th>#Param (M)</th><th>Sp</th><th>Se</th><th>S<sub>c</sub></th><th>Sp</th><th>Se</th><th>S<sub>c</sub></th></tr>",
         "</thead><tbody>"]
    cur_band = None
    for i, (k, b, c) in enumerate(rows):
        ref = c or b; m = meta[ref["family"]]
        name_fn, side_fn = _split_fns(b, c)
        used |= name_fn | side_fn["B"] | side_fn["C"]
        fams = " · ".join(x["family"] + ("" if x["metric_source"] in ("best", "final") else " · " + x["metric_source"]) for x in (b, c) if x)
        cite = f'<span class="run">\\cite{{{m["cite"].replace(" ", "")}}}{" · backbone " + m["backbone_cite"].replace(" ", "") if m.get("backbone_cite") else ""} · {fams}</span>'
        first = ""
        if row_band[i] != cur_band:
            cur_band = row_band[i]
            first = f'<td class="pm-band" rowspan="{row_band.count(cur_band)}"><span>{BAND_SHORT[cur_band]}</span></td>'
        cells = [first, f'<td class="name">{_html_esc(m["method"])}{_sup(name_fn, html=True)}{cite}</td>', f"<td>{_html_esc(tex2html(m['arch']))}</td>", f'<td class="m">{_param_cell(b, c, params)}</td>']
        for t, a, other in (("B", b, c), ("C", c, b)):
            if a is None:
                na = (other.get("na_fn") or "") if other else ""
                fb = fallback.get((t, (other.get("na_fallback") or "") if other else ""))
                if fb is not None:
                    agg = aggregate(fb, fam)
                    cells += [f'<td class="m">{_ours_html(*agg["sp"])}</td>', f'<td class="m">{_ours_html(*agg["se"])}</td>',
                              f'<td class="m">{_ours_html(*agg["score"])}</td>']
                    continue
                cells.append('<td class="m" colspan="3">' + ("n/a" if na else "—") + "</td>")
                continue
            agg = agg_of(a, t)
            cells += [f'<td class="m">{_ours_html(*agg["sp"], marks[t, "sp"].get(i))}</td>', f'<td class="m">{_ours_html(*agg["se"], marks[t, "se"].get(i))}</td>',
                      f'<td class="m">{_ours_html(*agg["score"], marks[t, "score"].get(i))}{_sup(side_fn[t], html=True)}</td>']
        L.append("<tr>" + "".join(cells) + "</tr>")
    L.append("</tbody></table></div>")
    notes = [f"<sup>{k}</sup> " + _html_esc(tex2html(footnotes[FN_KEY["M"] + k])) for k in sorted((x for x in used if x.isdigit()), key=int)]
    L.append('<p class="pm-notes">' + " ".join(notes) + "</p></div>")
    return "\n".join(L) + "\n"


def _html_esc(x: str) -> str:
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ours_html(m, s, kind=None):
    if m is None:
        return "—"
    core = f"{m:.2f}"
    core = {"best": f"<b>{core}</b>", "second": f"<u>{core}</u>"}.get(kind, core)
    return core + (f'<span class="sd">±{s:.2f}</span>' if s is not None else "")


def patchmix_html(anchors, fam, meta, footnotes):
    """종합 대장(아티팩트) 미리보기: table1/2/3 과 같은 행·수치."""
    out = []
    tex2html = lambda t: (t.replace(r"$\pm$", "±").replace(r"$\Delta$", "Δ").replace(r"$-$", "−").replace(r"$\to$", "→").replace(r"$\ge$", "≥")
                          .replace(r"$\alpha$", "α").replace(r"\ ", " ").replace("$\\times$", "×"))
    titles = {"A": "table1_ICHBI_Diff · ICBHI 2017 공식 60/40 · 4-class", "B": "table2_SPRSound_Diff · SPRSound 2022 inter-patient · 7-class",
              "C": "table4_OOD · ICBHI 학습 → SPRSound inter-patient 제로샷 (7→4 병합, 1,429 event)"}
    for table in ("A",):
        all_rows = [a for a in anchors if a["table"] == table]
        aggs = {a["family"] + "|" + a["metric_source"]: aggregate(a, fam) for a in all_rows}
        rows = sorted((a for a in all_rows if a.get("pm_order")), key=lambda a: (BAND_ORDER.index(a["band"]), float(a["pm_order"])))
        key = lambda a: a["family"] + "|" + a["metric_source"]
        comp = lambda a: meta[a["family"]]["comparable"] == "1"
        marks = {k: _rank_marks([aggs[key(a)][k][0] if (comp(a) and aggs[key(a)]["n"]) else None for a in rows])
                 for k in ("sp", "se", "score")}
        used = set()
        L = [f'<div class="pm pm-{table}">', f'<p class="pm-title"><b>{titles[table]}</b> · {len(rows)} 행</p>', '<div class="tbl"><table class="pmt"><thead>',
             '<tr class="grp"><th colspan="5"></th><th colspan="3">reported in paper</th><th colspan="3">ours (5 seeds)</th><th></th></tr>',
             "<tr><th></th><th>method</th><th>architecture</th><th>pretrain</th><th>venue</th><th>Sp</th><th>Se</th><th>S<sub>c</sub></th><th>Sp</th><th>Se</th><th>S<sub>c</sub></th><th>ΔS<sub>c</sub></th></tr>",
             "</thead><tbody>"]
        cur_band = None
        for i, a in enumerate(rows):
            m = meta[a["family"]]; agg = aggs[key(a)]
            fns = [x for x in a["fn"].split(";") if x]
            if table == "A":
                fns = sorted(_local(fns, "A"), key=int)
            used.update(fns)
            fn = f"<sup>{','.join(fns)}</sup>" if fns else ""
            cite = f'<span class="run">\\cite{{{m["cite"].replace(" ", "")}}} · {a["family"]}{" · " + a["metric_source"] if table == "C" else ""}</span>'
            first = ""
            if a["band"] != cur_band:
                cur_band = a["band"]
                first = f'<td class="pm-band" rowspan="{sum(1 for b in rows if b["band"] == cur_band)}"><span>{BAND_SHORT[cur_band]}</span></td>'
            cells = [first, f'<td class="name">{_html_esc(m["method"])}{fn}{cite}</td>', f"<td>{_html_esc(tex2html(m['arch']))}</td>",
                     f"<td>{_html_esc(m['pretrain'])}</td>", f"<td><i>{_html_esc(m['venue'].replace(chr(96), chr(8217)))}</i></td>",
                     f'<td class="m">{_rep(a["paper_sp"])}</td>', f'<td class="m">{_rep(a["paper_se"])}</td>', f'<td class="m">{_rep(a["paper_score"])}</td>',
                     f'<td class="m">{_ours_html(*agg["sp"], marks["sp"].get(i))}</td>', f'<td class="m">{_ours_html(*agg["se"], marks["se"].get(i))}</td>',
                     f'<td class="m">{_ours_html(*agg["score"], marks["score"].get(i))}</td>']
            if not a["paper_score"]:
                cells.append('<td class="m"></td>')
            elif a["delta"] != "score" or not agg["n"]:
                cells.append('<td class="m"><span class="d z">—</span></td>')
            else:
                dd = agg["score"][0] - float(a["paper_score"])
                cls = "p" if dd >= 0 else ("nn" if dd <= -2.5 else "n")
                cells.append(f'<td class="m"><span class="d {cls}">{delta_str(dd, tex=False)}</span></td>')
            L.append("<tr>" + "".join(cells) + "</tr>")
        L.append("</tbody></table></div>")
        key = (lambda k: FN_KEY["A"] + k) if table == "A" else (lambda k: k)
        notes = [f"<sup>{k}</sup> " + _html_esc(tex2html(footnotes[key(k)])) for k in sorted((x for x in used if x.isdigit()), key=int)]
        L.append('<p class="pm-notes">' + " ".join(notes) + "</p></div>")
        out.append("\n".join(L))
    out.append(merged_html(anchors, fam, meta, footnotes, tex2html))
    return "\n".join(out) + "\n"

def numbers_md(anchors, fam, aggs_by_table, git_sha):
    out = ["# 논문 수치 팩 (자동 생성 — 손으로 고치지 말 것)", "",
           f"생성 {time.strftime('%Y-%m-%d %H:%M')} · results_master {MASTER.stat().st_size} B, mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(MASTER.stat().st_mtime))} · git {git_sha}",
           "표기 Sp / Se / Score, seed 평균±표본표준편차. Δ = 우리 Score − 논문 Score, Δ/SE 는 우리 std/√n 기준.", ""]
    mismatches = []
    for table, title in (("A", "A · ICBHI → ICBHI"), ("B", "B · SPRSound → SPRSound"), ("C", "C · ICBHI → SPRSound (zero-shot OOD)")):
        out += [f"## {title}", "", "| # | 모델 | band | 공개 | family | n | Sp | Se | Score | 논문 (Sp/Se/Score±) | Δ | Δ/SE | 판정 | 부가 | 비고 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in sorted((x for x in anchors if x["table"] == table), key=lambda x: int(x["order"])):
            agg = aggs_by_table[table][a["family"] + "|" + a["metric_source"]]
            d = dse = None
            if a["delta"] == "score" and a["paper_score"] and agg["n"]:
                d = agg["score"][0] - float(a["paper_score"])
                se = agg["score"][1] / math.sqrt(agg["n"]) if agg["n"] > 1 else None
                dse = d / se if se else None
            extra = []
            if agg["extra"].get("binary_score"):
                extra.append(f"논문 산식 이진 {fmt(*agg['extra']['binary_score'], tex=False)} (Se {fmt(*agg['extra']['binary_se'], tex=False)})")
            if agg["extra"].get("best_on_test"):
                extra.append(f"best-on-test {fmt(*agg['extra']['best_on_test'], tex=False)}")
            if a["anchor_family"]:
                aa = aggregate({**a, "family": a["anchor_family"], "metric_source": "best"}, fam)
                extra.append(f"앵커 재현 `{a['anchor_family']}` {fmt(*aa['score'], tex=False)} (n={aa['n']})")
            flag = "" if agg["done_all"] else " ⚠ 미완주 포함"
            nflag = "" if agg["n"] == int(a["expect_n"]) else f" ⚠ 기대 {a['expect_n']}"
            ledger = fnum(a["ledger_score"])
            if ledger is not None and agg["n"] and abs(agg["score"][0] - ledger) > 0.01:
                mismatches.append((table, a["model"], a["family"], agg["score"][0], ledger))
                nflag += f" ❌ 원장 {ledger:.2f}"
            out.append("| " + " | ".join([a["order"], label_cell(a, tex=False), a["band"], a["disclosure"], f"`{a['family']}`",
                        f"{agg['n']}{nflag}{flag}", fmt(*agg["sp"], tex=False), fmt(*agg["se"], tex=False), fmt(*agg["score"], bold=True, tex=False),
                        paper_cell(a, "all", tex=False) + (f" ({a['paper_src']})" if a["paper_src"] else ""),
                        delta_str(d, tex=False), (f"{dse:+.2f}" if dse is not None else "—"), a["verdict"], "; ".join(extra) or "—", a["note"]]) + " |")
        out.append("")
    out += ["## 자체 검사 (계산값 vs 원장 docs/PAPER_TABLE_DECISIONS_2026-09-07.md)", ""]
    if mismatches:
        out += [f"❌ 불일치 {len(mismatches)}건:"] + [f"- {t} {m} `{f}`: 계산 {c:.2f} vs 원장 {l:.2f}" for t, m, f, c, l in mismatches]
    else:
        out.append(f"✅ 전 행 일치 (|Δ| ≤ 0.01) — A {sum(1 for a in anchors if a['table']=='A')} · B {sum(1 for a in anchors if a['table']=='B')} · C {sum(1 for a in anchors if a['table']=='C')} 행")
    out += ["", "## 기호", "", "† 팀 논문 · ‡ 툴킷 변형 · 이 문서의 각주 번호는 anchors.fn 의 전역 id(docs/paper/footnotes.csv 1~14). 논문 표는 표별 로컬 번호(t1.1~4, t2.1~5; FN_MAP)로 다시 매긴다"]
    return "\n".join(out) + "\n", mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["patchmix", "bands"], default="patchmix",
                    help="patchmix(기본): table1_ICHBI_Diff (reported | ours | ΔSc) + table2_SPRSound (SPR→SPR | ICBHI→SPR 제로샷, ours 만) + numbers.md + HTML 미리보기. "
                         "bands: 옛 Method/Paper/Ours/Δ 밴드 표(--band/--metrics 변형) — 파일은 저장소에 두지 않는다")
    ap.add_argument("--band", choices=["family", "disclosure"], default="family")
    ap.add_argument("--metrics", choices=["score", "all"], default="all")
    ap.add_argument("--out", default=str(ROOT / "docs" / "paper"))
    args = ap.parse_args()
    anchors = list(csv.DictReader(open(ANCHORS)))
    footnotes = {r["id"]: r["text"] for r in csv.DictReader(open(FOOTNOTES))}
    _, fam = load_master()
    out = Path(args.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = "?"
    aggs_by_table = {}
    if args.style == "bands":
        for table, name in (("A", "table_icbhi"), ("B", "table_sprsound"), ("C", "table_ood")):
            tex, aggs = build_table(anchors, fam, table, args.band, args.metrics, footnotes)
            aggs_by_table[table] = aggs
            (out / "tables" / f"{name}_bands_{args.band}_{args.metrics}.tex").write_text(
                f"% generated by scripts/paper/build_paper_tables.py --style bands --band {args.band} --metrics {args.metrics} (git {sha}); do not edit\n" + tex)
    else:
        meta = load_meta()
        hdr = (f"% generated by scripts/paper/build_paper_tables.py (git {sha}); do not edit\n"
               "% requires: booktabs, multirow, graphicx (\\resizebox), xcolor (\\textcolor for ΔSc); cite keys = user's bib (docs/paper/model_meta.csv `cite`; backbone papers are cited in the text, listed in `backbone_cite` and as % comments)\n")
        tex, aggs_by_table["A"] = build_patchmix(anchors, fam, "A", meta, footnotes)
        (out / "tables" / f"{PM_TABLES['A'][0]}.tex").write_text(hdr + tex)
        tex, aggs_by_table["B"], aggs_by_table["C"] = build_merged(anchors, fam, meta, footnotes)
        (out / "tables" / f"{MERGED[0]}.tex").write_text(hdr + tex)
        (out / "tables" / "preview_patchmix.html").write_text(patchmix_html(anchors, fam, meta, footnotes))
    md, mism = numbers_md(anchors, fam, aggs_by_table, sha)
    (out / "numbers.md").write_text(md)
    print(f"style={args.style} → {out/'tables'} ({', '.join([v[0] for v in PM_TABLES.values()] + [MERGED[0]]) if args.style == 'patchmix' else 'bands'}) · numbers → {out/'numbers.md'} · mismatches {len(mism)}")
    for m in mism:
        print("  MISMATCH", m)
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
