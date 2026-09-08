#!/usr/bin/env python3
"""
논문 표 생성기 (제출 대장 B3/B4) — 손으로 옮기지 않는다.

입력: docs/results_master.csv (run 단위 수치) + docs/paper/anchors.csv (표 행 정의·논문값·기재 결정)
      + docs/paper/footnotes.csv (각주 문안) + outputs/<group>/<run>/ood.json (C 표)
출력: docs/paper/tables/table_icbhi.tex · table_sprsound.tex · table_ood.tex (booktabs)
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
    return core + (r"{\scriptsize\,$\pm$" + f"{s:.2f}" + "}" if pm and s is not None else "")


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
    "B": ("table2_SPRSound_Diff", "SPRSound 2022 (inter-patient test set, 7-class cycle classification)", "sprsound_diff"),
    "C": ("table4_OOD", "zero-shot transfer: ICBHI-trained checkpoints evaluated on the SPRSound inter-patient test set (7$\\to$4 merged labels, 1,429 events)", "ood"),
}


def _rep(x):
    """논문 보고값 셀: 없으면 빈칸(사용자 지시), 있으면 소수 2자리"""
    return "" if x in ("", None) else num2(x)


def build_patchmix(anchors, fam, table, meta, footnotes):
    """사용자 형식(2026-09-08): method · architecture · pretrain · venue | reported Sp/Se/Sc | ours Sp/Se/Sc (5-seed mean±std) | ΔSc(색).
    행 = anchors.pm_order 가 있는 행, 그 순서. 논문값이 없는 행은 reported·ΔSc 를 비운다. 같은 프로토콜이 아니면 ΔSc = --.
    best/second 는 comparable=1 행끼리, 우리 값 기준. 반환 aggs 는 표의 전 행(numbers.md 용)."""
    all_rows = [a for a in anchors if a["table"] == table]
    aggs = {a["family"] + "|" + a["metric_source"]: aggregate(a, fam) for a in all_rows}
    rows = sorted((a for a in all_rows if a.get("pm_order")), key=lambda a: (BAND_ORDER.index(a["band"]), int(a["pm_order"])))
    def agg_of(a): return aggs[a["family"] + "|" + a["metric_source"]]
    def comp(a): return meta[a["family"]]["comparable"] == "1"
    marks = {k: _rank_marks([agg_of(a)[k][0] if (comp(a) and agg_of(a)["n"]) else None for a in rows])
             for k in ("sp", "se", "score")}
    stem, dsdesc, label = PM_TABLES[table]
    colspec = "p{2pt}llcl" + "ccc" + "ccc" + "c"
    head = [r"& \multirow{2}{*}{method} & \multirow{2}{*}{architecture} & \multirow{2}{*}{pretrain} & \multirow{2}{*}{venue} & "
            r"\multicolumn{3}{c}{reported in paper} & \multicolumn{3}{c}{ours (5 seeds)} & \multirow{2}{*}{$\Delta$Sc} \\",
            r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
            r"& & & & & $S_p$ & $S_e$ & Sc & $S_p$ & $S_e$ & \textbf{Sc} & \\"]
    common = (r" ``Reported'' copies the numbers from each paper (single run or mean over the paper's seeds; blank where the paper reports "
              r"nothing for this setting); ``ours'' is the toolkit re-implementation with five seeds (mean\,$\pm$\,std) under one protocol. "
              r"$\Delta$Sc = ours $-$ reported Score (\textcolor{dpos}{green}: at or above, \textcolor{dneg}{red}: below), shown only where the "
              r"two are measured under the same protocol. \textbf{Best} and \underline{second best} (ours) among directly comparable rows.")
    common = common + r" Rows are grouped by the type of contribution (rotated labels)."
    cap = {"A": "Reported vs.\\ reproduced results on " + dsdesc + ".",
           "B": "Reported vs.\\ reproduced results on " + dsdesc + ".",
           "C": "Reported vs.\\ reproduced " + dsdesc + "."}[table] + common
    L = [DELTA_COLORS, r"\begin{table*}[!t]", r"\centering", r"\caption{" + cap + "}", r"\label{tab:" + label + "}",
         r"\renewcommand{\arraystretch}{1}", r"\addtolength{\tabcolsep}{2pt}",
         r"\resizebox{\linewidth}{!}{", r"\begin{tabular}{" + colspec + "}", r"\toprule", *head, r"\hline \midrule"]
    used_fn = set()
    band_sizes = {b: sum(1 for a in rows if a["band"] == b) for b in BAND_ORDER}
    cur_band = None
    for i, a in enumerate(rows):
        agg = agg_of(a); m = meta[a["family"]]
        fns = [x for x in a["fn"].split(";") if x]
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
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    notes = [r"\textsuperscript{" + k + "} " + footnotes[k] for k in sorted((x for x in used_fn if x.isdigit()), key=int)]
    if notes:
        L += [r"\par\vspace{2pt}\begin{minipage}{\linewidth}\scriptsize", " ".join(notes), r"\end{minipage}"]
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n", aggs

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
    for table in ("A", "B", "C"):
        all_rows = [a for a in anchors if a["table"] == table]
        aggs = {a["family"] + "|" + a["metric_source"]: aggregate(a, fam) for a in all_rows}
        rows = sorted((a for a in all_rows if a.get("pm_order")), key=lambda a: (BAND_ORDER.index(a["band"]), int(a["pm_order"])))
        key = lambda a: a["family"] + "|" + a["metric_source"]
        comp = lambda a: meta[a["family"]]["comparable"] == "1"
        marks = {k: _rank_marks([aggs[key(a)][k][0] if (comp(a) and aggs[key(a)]["n"]) else None for a in rows])
                 for k in ("sp", "se", "score")}
        used = set()
        L = [f'<div class="pm pm-{table}">', f'<p class="pm-title"><b>{titles[table]}</b> · {len(rows)} 행</p>', '<div class="tbl"><table class="pmt"><thead>',
             '<tr class="grp"><th colspan="5"></th><th colspan="3">reported in paper</th><th colspan="3">ours (5 seeds)</th><th></th></tr>',
             "<tr><th></th><th>method</th><th>architecture</th><th>pretrain</th><th>venue</th><th>Sp</th><th>Se</th><th>Sc</th><th>Sp</th><th>Se</th><th>Sc</th><th>ΔSc</th></tr>",
             "</thead><tbody>"]
        cur_band = None
        for i, a in enumerate(rows):
            m = meta[a["family"]]; agg = aggs[key(a)]
            fns = [x for x in a["fn"].split(";") if x]; used.update(fns)
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
        notes = [f"<sup>{k}</sup> " + _html_esc(tex2html(footnotes[k])) for k in sorted((x for x in used if x.isdigit()), key=int)]
        L.append('<p class="pm-notes">' + " ".join(notes) + "</p></div>")
        out.append("\n".join(L))
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
    out += ["", "## 기호", "", "† 팀 논문 · ‡ 툴킷 변형 · 숫자 각주는 docs/paper/footnotes.csv (1 코드 없음 · 2 하이퍼파라미터 미제공 · 3 스플릿 상이 …)"]
    return "\n".join(out) + "\n", mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["patchmix", "bands"], default="patchmix",
                    help="patchmix(기본): table1_ICHBI_Diff · table2_SPRSound_Diff · table3_OOD (reported | ours | ΔSc) + numbers.md + HTML 미리보기. "
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
        for table in ("A", "B", "C"):
            tex, aggs = build_patchmix(anchors, fam, table, meta, footnotes)
            aggs_by_table[table] = aggs
            (out / "tables" / f"{PM_TABLES[table][0]}.tex").write_text(
                f"% generated by scripts/paper/build_paper_tables.py (git {sha}); do not edit\n"
                "% requires: booktabs, multirow, graphicx (\\resizebox), xcolor (\\textcolor for ΔSc); cite keys = user's bib (docs/paper/model_meta.csv `cite`)\n" + tex)
        (out / "tables" / "preview_patchmix.html").write_text(patchmix_html(anchors, fam, meta, footnotes))
    md, mism = numbers_md(anchors, fam, aggs_by_table, sha)
    (out / "numbers.md").write_text(md)
    print(f"style={args.style} → {out/'tables'} ({', '.join(v[0] for v in PM_TABLES.values()) if args.style == 'patchmix' else 'bands'}) · numbers → {out/'numbers.md'} · mismatches {len(mism)}")
    for m in mism:
        print("  MISMATCH", m)
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
