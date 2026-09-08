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

DISCLOSURE_LABEL = {"full": "Code + training scripts released", "code": "Code released",
                    "partial": "Partial release / paper-code mismatch", "none": "No code released"}
DISCLOSURE_ORDER = ["full", "code", "partial", "none"]
MARK = {"team": r"$^{\dagger}$", "assume": r"$^{\ast}$", "variant": r"$^{\ddagger}$"}


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
    score = sc + ((r" $\pm$ " if tex else "±") + sd if sd else "")
    if metrics == "score":
        return score
    if sp and se:
        return f"{sp} / {se} / {score}"
    return ("-- / -- / " if tex else "— / — / ") + score


def ours_cell(agg, metrics, tex=True):
    if agg["n"] == 0:
        return "--" if tex else "—"
    sc = fmt(*agg["score"], bold=True, tex=tex)
    if metrics == "score":
        return sc
    return f"{fmt(*agg['sp'], tex=tex)} / {fmt(*agg['se'], tex=tex)} / {sc}"


def label_cell(a, tex=True):
    marks = "".join(MARK[f] for f in a["flags"].split(";") if f) if tex else \
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
    out += ["", "## 기호", "", "† 팀 논문 · \\* 가정 런(학습 명세 미공개) · ‡ 툴킷 변형 · 숫자 각주는 docs/paper/footnotes.csv"]
    return "\n".join(out) + "\n", mismatches


def main():
    ap = argparse.ArgumentParser()
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
    for table, name in (("A", "table_icbhi"), ("B", "table_sprsound"), ("C", "table_ood")):
        tex, aggs = build_table(anchors, fam, table, args.band, args.metrics, footnotes)
        aggs_by_table[table] = aggs
        suffix = "" if (args.band == "family" and args.metrics == "all") else f"_{args.band}_{args.metrics}"
        (out / "tables" / f"{name}{suffix}.tex").write_text(f"% generated by scripts/paper/build_paper_tables.py --band {args.band} --metrics {args.metrics} (git {sha}); do not edit\n" + tex)
    md, mism = numbers_md(anchors, fam, aggs_by_table, sha)
    (out / "numbers.md").write_text(md)
    print(f"tables → {out/'tables'} · numbers → {out/'numbers.md'} · mismatches {len(mism)}")
    for m in mism:
        print("  MISMATCH", m)
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
