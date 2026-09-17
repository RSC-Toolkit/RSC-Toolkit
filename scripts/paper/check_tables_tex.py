# -*- coding: utf-8 -*-
"""생성된 표 .tex 의 정적 점검 (LaTeX 툴체인 없이): 열 수 = colspec, \\textsuperscript 번호 ↔ 각주 정의 일치, 중괄호 균형, 본문 비ASCII 문자.

  .venv/bin/python scripts/paper/check_tables_tex.py [docs/paper/tables/table1_ICHBI_Diff.tex ...]
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT = [ROOT / "docs/paper/tables/table1_ICHBI_Diff.tex", ROOT / "docs/paper/tables/table2_SPRSound.tex"]


def colspec_count(spec: str) -> int:
    spec = re.sub(r"p\{[^}]*\}", "p", spec)          # p{2pt} → p
    spec = re.sub(r"@\{[^}]*\}|\|", "", spec)
    return sum(1 for ch in spec if ch in "lcrp")


def row_cells(row: str) -> int:
    body = re.sub(r"\\\\\s*$", "", row.strip())
    cells = re.split(r"(?<!\\)&", body)
    extra = sum(int(n) - 1 for n in re.findall(r"\\multicolumn\{(\d+)\}", body))
    return len(cells) + extra


def check(path: pathlib.Path) -> list[str]:
    errs: list[str] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    text = "\n".join(l for l in lines if not l.lstrip().startswith("%"))
    if text.count("{") != text.count("}"):
        errs.append(f"중괄호 불균형: {{ {text.count('{')} vs }} {text.count('}')}")
    bad = [(i + 1, l) for i, l in enumerate(lines) if not l.lstrip().startswith("%") and re.search(r"[^\x00-\x7F]", l)]
    for i, l in bad:
        errs.append(f"본문 비ASCII 문자 (line {i}): {l.strip()[:80]}")
    m = re.search(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}", text)   # p{2pt} 같은 1단계 중첩 허용
    if not m:
        return errs + ["tabular 없음"]
    ncol = colspec_count(m.group(1))
    in_tab = False; nrows = 0
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("%"):
            continue
        if "\\begin{tabular}" in s:
            in_tab = True; continue
        if "\\end{tabular}" in s:
            in_tab = False; continue
        if in_tab and s.endswith("\\\\"):
            nrows += 1; c = row_cells(s)
            if c != ncol:
                errs.append(f"열 수 {c} ≠ colspec {ncol} (line {i + 1}): {s[:90]}")
    body_sup = set(); fn_def = set()
    minipage = re.search(r"\\begin\{minipage\}.*?\\end\{minipage\}", text, re.S)
    if minipage:
        fn_def = set(re.findall(r"\\textsuperscript\{(\d+)\}", minipage.group(0)))
        body = text.replace(minipage.group(0), "")
    else:
        body = text
    for grp in re.findall(r"\\textsuperscript\{([\d,]+)\}", body):
        body_sup.update(grp.split(","))
    if body_sup - fn_def:
        errs.append(f"본문 표식에 정의 없는 각주: {sorted(body_sup - fn_def, key=int)}")
    if fn_def - body_sup:
        errs.append(f"정의만 있고 표식 없는 각주: {sorted(fn_def - body_sup, key=int)}")
    if fn_def and sorted(fn_def, key=int) != [str(i) for i in range(1, len(fn_def) + 1)]:
        errs.append(f"각주 번호가 1..n 연속이 아님: {sorted(fn_def, key=int)}")
    print(f"{path.name}: colspec {ncol} 열 · 데이터/헤더 행 {nrows} · 각주 {sorted(fn_def, key=int)} · 표식 {sorted(body_sup, key=int)}")
    return errs


def main() -> int:
    paths = [pathlib.Path(p) for p in sys.argv[1:]] or DEFAULT
    bad = 0
    for p in paths:
        errs = check(p)
        for e in errs:
            print("  !!", e); bad += 1
    print("OK" if not bad else f"{bad} 문제")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
