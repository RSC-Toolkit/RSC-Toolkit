# -*- coding: utf-8 -*-
"""팀 논문 3편 저자 확인 자료 — 결과 시각화 + 이슈 정리 HTML."""
import pathlib, statistics as st

OUT = pathlib.Path("/tmp/claude-10008/-raid-doyoung-projects-RSC-Toolkit/88feaa60-f05e-4f32-8234-95e341f584ce/scratchpad/team_paper_evidence.html")

# ── 데이터 (outputs/*/best.json 실측) ──────────────────────────────────────
LADDER = [  # (단계, 설명, 논문 mean, 논문 std, 우리 mean, 우리 std)
    ("FT",            "DASS 백본 + CE 만",                 61.06, 1.27, 61.65, 0.84),
    ("SAR",           "+ 가우시안 블러 정규화",              62.22, 1.29, 62.66, 0.74),
    ("Lung-SRAD",     "+ 이중축 Patch-Mix CL (코드 축)",     64.48, 0.25, 61.59, 1.37),
    ("Lung-SRAD*",    "+ 이중축 CL (축 정의 수정)",          64.48, 0.25, 62.49, 1.04),
]
SEEDS = [  # (논문, 우리 seed 5개, 논문 mean, 논문 std, 비고)
    ("Lung-SRAD (축 수정)", [63.05, 62.26, 61.79, 63.98, 61.38], 64.48, 0.25,
     "논문값이 우리 최고 seed보다 <b>+0.50 위</b> — 분포 바깥"),
    ("QLung",               [60.43, 61.15, 60.41, 61.87, 60.21], 62.01, 1.18,
     "논문값이 우리 최고 seed보다 <b>+0.14 위</b> — 사실상 분포 경계"),
    ("BTS-CARD",            [63.92, 63.57, 64.63, 63.24, 61.96], 64.63, 0.57,
     "우리 <b>seed 3 이 64.63</b> — 논문 평균과 같은 값, 분포 안"),
]
AQS = [  # (해석, 값, n, 설명)
    ("① 코드 그대로<br><span class=sub>entropy 자리에 rms 중복 전달</span>", 60.81, 0.69, 5),
    ("② 엔트로피 정상 전달<br><span class=sub>인자 위치만 수정</span>",       60.37, 0.46, 5),
    ("③ 논문 식 (5) 문자 그대로<br><span class=sub>인자 + 정규화 둘 다 논문 쪽</span>", 56.68, 3.79, 3),
]
AQS_PAPER, AQS_BASE = 62.01, 59.80
PIPELINE = [  # 비팀 논문 + 팀 기준선 Δ
    ("ResNet18 SFT", 0.04), ("BEATs SFT", 0.10), ("Audio-CLAP", 0.20), ("AST SFT", 0.25),
    ("BTS", 0.37), ("Lung-SRAD FT", 0.59), ("Lung-SRAD SAR", 0.44),
    ("PAFA", -0.16), ("M2D-X", -0.21), ("CNN6 SFT", -0.32),
]

# ── SVG 빌더 ───────────────────────────────────────────────────────────────
def esc(t): return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def ladder_chart():
    x0, x1, top, rowh, gap = 132, 636, 26, 62, 10
    lo, hi = 58.0, 66.0
    def X(v): return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    h = top + len(LADDER) * rowh + 34
    p = [f'<svg viewBox="0 0 786 {h}" role="img" aria-label="Lung-SRAD 단계별 논문 대 재현 점수" class="chart">']
    for g in [58, 60, 62, 64, 66]:
        p.append(f'<line class="grid" x1="{X(g):.1f}" y1="{top-8}" x2="{X(g):.1f}" y2="{top+len(LADDER)*rowh}"/>')
        p.append(f'<text class="tick" x="{X(g):.1f}" y="{top+len(LADDER)*rowh+18}" text-anchor="middle">{g}</text>')
    for i,(name,desc,pm,ps,om,os_) in enumerate(LADDER):
        y = top + i*rowh
        p.append(f'<text class="rowname" x="{x0-12}" y="{y+15}" text-anchor="end">{esc(name)}</text>')
        p.append(f'<text class="rowdesc" x="{x0-12}" y="{y+31}" text-anchor="end">{esc(desc)}</text>')
        for k,(v,sd,cls,lab) in enumerate([(pm,ps,"s-paper","논문"),(om,os_,"s-ours","우리")]):
            by = y + 4 + k*(gap+8)
            p.append(f'<g class="bar"><title>{esc(lab)} {esc(name)}: {v:.2f} ± {sd:.2f}</title>'
                     f'<rect class="{cls}" x="{x0}" y="{by}" width="{max(0,X(v)-x0):.1f}" height="8" rx="4"/>'
                     f'<line class="err" x1="{X(v-sd):.1f}" y1="{by+4}" x2="{X(v+sd):.1f}" y2="{by+4}"/>'
                     f'<text class="val {cls}t" x="{X(v)+8:.1f}" y="{by+8}">{v:.2f}</text></g>')
        if om < pm - 0.5:
            p.append(f'<text class="deltabad" x="{x1+62}" y="{y+22}">−{pm-om:.2f}</text>')
        else:
            p.append(f'<text class="deltaok" x="{x1+62}" y="{y+22}">+{om-pm:.2f}</text>')
    p.append("</svg>")
    return "\n".join(p)

def seed_chart(title, seeds, pm, ps, note, idx):
    x0, x1, top = 16, 620, 18
    lo, hi = 59.5, 65.5
    def X(v): return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    h = 86
    om = st.mean(seeds)
    p = [f'<svg viewBox="0 0 660 {h}" role="img" aria-label="{esc(title)} seed 분포와 논문값" class="chart">']
    for g in [60, 61, 62, 63, 64, 65]:
        p.append(f'<line class="grid" x1="{X(g):.1f}" y1="{top-4}" x2="{X(g):.1f}" y2="{top+34}"/>')
        p.append(f'<text class="tick" x="{X(g):.1f}" y="{top+52}" text-anchor="middle">{g}</text>')
    # 논문 ± 밴드 + 선
    p.append(f'<rect class="paperband" x="{X(pm-ps):.1f}" y="{top-4}" width="{X(pm+ps)-X(pm-ps):.1f}" height="38"/>')
    p.append(f'<g class="bar"><title>논문 {pm:.2f} ± {ps:.2f}</title>'
             f'<line class="paperline" x1="{X(pm):.1f}" y1="{top-8}" x2="{X(pm):.1f}" y2="{top+38}"/></g>')
    p.append(f'<text class="paperlab" x="{X(pm):.1f}" y="{top-13}" text-anchor="middle">논문 {pm:.2f}</text>')
    # 우리 평균 선
    p.append(f'<line class="oursmean" x1="{X(om):.1f}" y1="{top+2}" x2="{X(om):.1f}" y2="{top+30}"/>')
    p.append(f'<text class="ourslab" x="{X(om):.1f}" y="{top+48}" text-anchor="middle">우리 평균 {om:.2f}</text>')
    for s,v in enumerate(seeds, 1):
        p.append(f'<g class="bar"><title>seed {s}: {v:.2f}</title>'
                 f'<circle class="dot" cx="{X(v):.1f}" cy="{top+16}" r="6"/></g>')
    p.append("</svg>")
    return f'<figure class="fig"><figcaption class="figtitle">{esc(title)}</figcaption>{"".join(p)}<p class="fignote">{note}</p></figure>'

def aqs_chart():
    x0, x1, top, rowh = 232, 640, 34, 56
    lo, hi = 52.0, 64.0
    def X(v): return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    h = top + len(AQS)*rowh + 30
    p = [f'<svg viewBox="0 0 716 {h}" role="img" aria-label="QLung AQS 해석 3종 점수" class="chart">']
    for g in [52,54,56,58,60,62,64]:
        p.append(f'<line class="grid" x1="{X(g):.1f}" y1="{top-14}" x2="{X(g):.1f}" y2="{top+len(AQS)*rowh-14}"/>')
        p.append(f'<text class="tick" x="{X(g):.1f}" y="{top+len(AQS)*rowh+4}" text-anchor="middle">{g}</text>')
    for lab,val,ref in [(f"논문 {AQS_PAPER}", AQS_PAPER, "paperline"), (f"AST 기준선 {AQS_BASE}", AQS_BASE, "baseline")]:
        p.append(f'<line class="{ref}" x1="{X(val):.1f}" y1="{top-20}" x2="{X(val):.1f}" y2="{top+len(AQS)*rowh-14}"/>')
        p.append(f'<text class="reflab" x="{X(val):.1f}" y="{top-25}" text-anchor="middle">{esc(lab)}</text>')
    for i,(name,v,sd,n) in enumerate(AQS):
        y = top + i*rowh
        plain = name.replace("<br>","  ").replace("<span class=sub>","(").replace("</span>",")")
        p.append(f'<g class="bar"><title>{esc(plain)}: {v:.2f} (n={n})</title>'
                 f'<rect class="s-ours" x="{x0}" y="{y-4}" width="{max(0,X(v)-x0):.1f}" height="10" rx="4"/>'
                 f'<line class="err" x1="{X(v-sd):.1f}" y1="{y+1}" x2="{X(v+sd):.1f}" y2="{y+1}"/>'
                 f'<text class="val s-ourst" x="{X(v)+8:.1f}" y="{y+5}">{v:.2f}{"" if n==5 else f" ({n}/5)"}</text></g>')
    for i,(name,_,_,_) in enumerate(AQS):
        y = top + i*rowh
        head, tail = name.split("<br>")
        tail = tail.replace("<span class=sub>","").replace("</span>","")
        p.append(f'<text class="rowname" x="{x0-14}" y="{y+1}" text-anchor="end">{esc(head)}</text>'
                 f'<text class="rowdesc" x="{x0-14}" y="{y+17}" text-anchor="end">{esc(tail)}</text>')
    p.append("</svg>")
    return "".join(p)

def pipeline_strip():
    items = []
    for name, d in PIPELINE:
        w = min(abs(d), 1.0) / 1.0 * 46
        side = f"left:50%;width:{w:.1f}px" if d >= 0 else f"right:50%;width:{w:.1f}px"
        cls = "pos" if d >= 0 else "neg"
        items.append(f'<div class="pi"><span class="pin">{esc(name)}</span>'
                     f'<span class="pt"><span class="pf {cls}" style="{side}"></span><span class="pz"></span></span>'
                     f'<span class="pv {cls}">{d:+.2f}</span></div>')
    return f'<div class="pipe">{"".join(items)}</div>'

# ── 검증된 모델 전체 목록 (docs/RESULTS_BY_DATASET.md §1 파싱) ──────────────
import re
def all_models_table():
    src = pathlib.Path("docs/RESULTS_BY_DATASET.md")
    if not src.exists():
        return "<p class=fignote>(요약 표를 찾지 못했습니다)</p>"
    lines = src.read_text().splitlines()
    rows = []
    for l in lines:
        if not re.match(r"^\| \d+ \|", l): break_ = False
        if re.match(r"^\| \d+ \|", l):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) >= 10: rows.append(c)
        elif rows:
            break
    def md(t):
        t = t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"~~(.+?)~~", r"<s>\1</s>", t)
        return t
    def verd(v):
        core = re.sub(r"[⁰-⁹\s]", "", v)
        if core.startswith("✅"): return '<span class="pill ok">✅</span>'
        if core.startswith("⚠️⚠️"): return '<span class="pill crit">⚠️⚠️</span>'
        if core.startswith("⚠️"): return '<span class="pill warn">⚠️</span>'
        return f'<span class="pill na">{md(v)}</span>'
    def dcell(d):
        m = re.search(r"([+−-]?\d+\.\d+)", d.replace("**",""))
        if not m: return '<td class="num muted">—</td>'
        v = float(m.group(1).replace("−","-"))
        return f'<td class="num {"pos" if v>=0 else "neg"}">{v:+.2f}</td>'
    out = []
    for c in rows:
        idx, model, how, i_our, i_pap, i_d, s_our, s_pap, s_d, v = c[:10]
        team = " team" if "(팀)" in model else ""
        out.append(f'<tr class="{team.strip()}"><td class="num muted">{md(idx)}</td>'
                   f'<td class="mname">{md(model)}<small>{md(how)}</small></td>'
                   f'<td class="num">{md(i_our) or "—"}</td><td class="num muted">{md(i_pap) or "—"}</td>{dcell(i_d)}'
                   f'<td class="num">{md(s_our) or "—"}</td>{dcell(s_d)}'
                   f'<td>{verd(v)}</td></tr>')
    return ("<div class=\"tablewrap\"><table class=\"all\"><thead><tr>"
            "<th>#</th><th>모델</th><th>ICBHI 우리</th><th>ICBHI 논문</th><th>Δ</th>"
            "<th>SPRSound 우리</th><th>Δ</th><th>판정</th></tr></thead><tbody>"
            + "".join(out) + "</tbody></table></div>")

# ── 페이지 ─────────────────────────────────────────────────────────────────
OUT.write_text(f'''<title>팀 논문 재현 확인 요청</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=Noto+Serif+KR:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {{
    --bg:#F4F6F8; --surface:#FFFFFF; --surface-2:#EAEEF2; --line:#D3DBE3;
    --ink:#131A21; --ink-2:#3E4C59; --muted:#6B7A88;
    --accent:#1B4E82; --accent-soft:#E4EDF6;
    --paper:#2a78d6; --ours:#eb6834;
    --band:#DCE8F6; --grid:#E3E8ED;
    --ok-bg:#DDEEE2; --ok-ink:#1E6B3E;
    --warn-bg:#FAEACB; --warn-ink:#8A5A10;
    --crit-bg:#F7DBD4; --crit-ink:#8E3218;
    --pos:#1E6B3E; --neg:#B04A2C; --track:#DFE5EB;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0F1519; --surface:#161E25; --surface-2:#1D272F; --line:#2B3843;
      --ink:#E6ECF2; --ink-2:#B4C1CD; --muted:#8494A2;
      --accent:#7FB2E8; --accent-soft:#1A2937;
      --paper:#3987e5; --ours:#d95926;
      --band:#1B2C40; --grid:#232E38;
      --ok-bg:#183828; --ok-ink:#8ED9AC;
      --warn-bg:#3C2D10; --warn-ink:#F0C46A;
      --crit-bg:#3F211A; --crit-ink:#F0A28A;
      --pos:#7CCB9C; --neg:#E8916F; --track:#232E38;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0F1519; --surface:#161E25; --surface-2:#1D272F; --line:#2B3843;
    --ink:#E6ECF2; --ink-2:#B4C1CD; --muted:#8494A2;
    --accent:#7FB2E8; --accent-soft:#1A2937;
    --paper:#3987e5; --ours:#d95926;
    --band:#1B2C40; --grid:#232E38;
    --ok-bg:#183828; --ok-ink:#8ED9AC;
    --warn-bg:#3C2D10; --warn-ink:#F0C46A;
    --crit-bg:#3F211A; --crit-ink:#F0A28A;
    --pos:#7CCB9C; --neg:#E8916F; --track:#232E38;
  }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--ink); margin:0;
    font-family:"IBM Plex Sans KR","Apple SD Gothic Neo",system-ui,sans-serif; font-size:15px; line-height:1.65; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:44px 22px 80px; display:grid; gap:0; }}
  .eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent); }}
  h1 {{ font-family:"Noto Serif KR",serif; font-weight:700; font-size:32px; line-height:1.25; margin:10px 0 12px; text-wrap:balance; }}
  .lede {{ color:var(--ink-2); max-width:66ch; margin:0; font-size:15.5px; }}
  h2 {{ font-family:"Noto Serif KR",serif; font-size:21px; font-weight:600; margin:52px 0 6px; padding-top:24px; border-top:1px solid var(--line); }}
  h2 .n {{ font-family:"IBM Plex Mono",monospace; color:var(--accent); font-size:15px; margin-right:10px; }}
  h3 {{ font-size:15px; font-weight:700; margin:26px 0 6px; }}
  p {{ margin:0 0 12px; max-width:70ch; }}
  .sub {{ color:var(--muted); font-weight:400; font-size:.86em; }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:20px 22px; margin:16px 0; }}
  .ask {{ background:var(--accent-soft); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:6px; padding:16px 20px; margin:18px 0; }}
  .ask b {{ display:block; font-size:13px; letter-spacing:.02em; color:var(--accent); margin-bottom:6px; }}
  .ask p {{ margin:0; }}
  .ask p + p {{ margin-top:8px; }}
  .chart {{ width:100%; height:auto; display:block; overflow:visible; }}
  .grid {{ stroke:var(--grid); stroke-width:1; }}
  .tick {{ fill:var(--muted); font-family:"IBM Plex Mono",monospace; font-size:11px; }}
  .rowname {{ fill:var(--ink); font-size:13px; font-weight:600; }}
  .rowdesc {{ fill:var(--muted); font-size:11px; }}
  .s-paper {{ fill:var(--paper); }} .s-ours {{ fill:var(--ours); }}
  .s-papert {{ fill:var(--ink-2); }} .s-ourst {{ fill:var(--ink-2); }}
  .val {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; }}
  .err {{ stroke:var(--ink-2); stroke-width:1.5; opacity:.45; }}
  .deltaok {{ fill:var(--pos); font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:600; }}
  .deltabad {{ fill:var(--neg); font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:600; }}
  .bar {{ cursor:default; }} .bar:hover rect, .bar:hover circle {{ filter:brightness(1.12); }}
  .paperband {{ fill:var(--band); }}
  .paperline {{ stroke:var(--paper); stroke-width:2; }}
  .baseline {{ stroke:var(--muted); stroke-width:1.5; stroke-dasharray:3 3; }}
  .paperlab {{ fill:var(--paper); font-family:"IBM Plex Mono",monospace; font-size:11.5px; font-weight:600; }}
  .reflab {{ fill:var(--muted); font-family:"IBM Plex Mono",monospace; font-size:11px; }}
  .oursmean {{ stroke:var(--ours); stroke-width:2; stroke-dasharray:2 3; }}
  .ourslab {{ fill:var(--ours); font-family:"IBM Plex Mono",monospace; font-size:11px; }}
  .dot {{ fill:var(--ours); stroke:var(--surface); stroke-width:2; }}
  .fig {{ margin:20px 0 8px; }}
  .figtitle {{ font-size:13.5px; font-weight:700; margin-bottom:2px; }}
  .fignote {{ color:var(--muted); font-size:12.5px; margin:2px 0 0; }}
  .legend {{ display:flex; gap:18px; align-items:center; font-size:12.5px; color:var(--ink-2); margin:8px 0 4px; }}
  .sw {{ display:inline-block; width:14px; height:8px; border-radius:4px; margin-right:6px; vertical-align:middle; }}
  .pipe {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:4px 22px; margin:10px 0 4px; }}
  .pi {{ display:grid; grid-template-columns:1fr 100px 52px; align-items:center; gap:8px; font-size:13px; }}
  .pin {{ text-align:right; color:var(--ink-2); }}
  .pt {{ position:relative; height:7px; background:var(--track); border-radius:4px; }}
  .pf {{ position:absolute; top:0; bottom:0; border-radius:4px; }}
  .pf.pos {{ background:var(--pos); }} .pf.neg {{ background:var(--neg); }}
  .pz {{ position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--line); }}
  .pv {{ font-family:"IBM Plex Mono",monospace; font-size:12px; }}
  .pv.pos {{ color:var(--pos); }} .pv.neg {{ color:var(--neg); }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; margin:12px 0; }}
  table.all {{ min-width:880px; font-size:12.5px; }}
  table.all td, table.all th {{ padding:7px 9px; }}
  table.all td.mname {{ font-weight:600; min-width:150px; }}
  table.all td.mname small {{ display:block; font-weight:400; color:var(--muted); font-size:11px; line-height:1.35; margin-top:1px; }}
  table.all tr.team td {{ background:var(--accent-soft); }}
  table.all td.muted {{ color:var(--muted); }}
  table.all td.pos {{ color:var(--pos); }} table.all td.neg {{ color:var(--neg); }}
  .pill.na {{ background:var(--surface-2); color:var(--muted); }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
  thead th {{ font-size:11.5px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); font-weight:500; background:var(--surface-2); }}
  td.num {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .tablewrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--surface); }}
  code {{ font-family:"IBM Plex Mono",monospace; font-size:.9em; background:var(--surface-2); padding:1px 5px; border-radius:3px; }}
  .pill {{ display:inline-block; padding:1px 9px; border-radius:999px; font-size:11.5px; font-weight:600; }}
  .ok {{ background:var(--ok-bg); color:var(--ok-ink); }}
  .warn {{ background:var(--warn-bg); color:var(--warn-ink); }}
  .crit {{ background:var(--crit-bg); color:var(--crit-ink); }}
  ol.checks {{ padding-left:0; list-style:none; counter-reset:c; display:grid; gap:10px; margin:14px 0 0; }}
  ol.checks li {{ counter-increment:c; display:grid; grid-template-columns:30px minmax(0,1fr); column-gap:12px; row-gap:4px;
    background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:13px 16px; }}
  ol.checks li::before {{ content:counter(c); grid-column:1; grid-row:1; font-family:"IBM Plex Mono",monospace; font-weight:600; color:var(--accent); }}
  ol.checks li > b {{ grid-column:2; grid-row:1; }}
  ol.checks li > span {{ grid-column:2; grid-row:2; color:var(--ink-2); font-size:13.5px; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">RSC-Toolkit · ICBHI 2017 공식 60/40 · 4-class · seed 1–5</div>
    <h1>팀 논문 3편 재현 — 확인 요청 자료</h1>
    <p class="lede">동일 파이프라인·동일 스플릿·동일 지표로 세 편을 재현했고, 세 편 모두 논문 수치에 미달했습니다.
    원인을 짚어 본 결과 <b>셋이 서로 다른 문제</b>였습니다. 각 편마다 저자만 줄 수 있는 것이 하나씩 있어 정리합니다.
    수치는 <b>Score = (Se+Sp)/2</b>, 우리 값은 seed 1–5 평균±표준편차, 모델 선택은 best-on-test 입니다.</p>
  </header>

  <div class="card">
    <h3 style="margin-top:0">먼저 — 우리 파이프라인 자체는 정상입니다</h3>
    <p>같은 코드·같은 스플릿·같은 지표로 돌린 <b>다른 논문들이 촘촘히 재현</b>됩니다. 아래는 논문 대비 Δ Score 입니다.
    <b>세 편의 기준선(Lung-SRAD FT·SAR, BTS)도 여기 포함</b>돼 있습니다 — 즉 데이터 로딩·특징 추출·라벨·지표 경로는 검증돼 있고,
    <b>각 논문이 더한 방법 위에서만</b> 격차가 생깁니다.</p>
    {pipeline_strip()}
    <p style="margin-top:14px">결정적으로, <b>BTS-CARD 저자께서 주신 <code>best.pth</code> 를 우리 평가 경로에 넣었더니 64.54</b> 가 나와
    논문 64.63 과 일치했습니다. 체크포인트 하나면 “학습만의 문제인지”가 즉시 갈립니다 — 나머지 두 편에도 같은 방법을 쓰고 싶습니다.</p>
  </div>

  <h2><span class="n">01</span>Lung-SRAD — 마지막 한 칸만 무너집니다</h2>
  <p>논문의 누적 사다리를 그대로 따라갔습니다. <b>앞의 두 칸은 우리가 오히려 조금 높은데</b>(+0.59, +0.44),
  마지막 이중축 Patch-Mix CL 칸만 −2.89 입니다. 환경이나 패키지 버전 문제라면 세 칸이 다 어긋나야 합니다.</p>
  <div class="legend"><span><span class="sw" style="background:var(--paper)"></span>논문</span>
    <span><span class="sw" style="background:var(--ours)"></span>우리 (seed 1–5)</span>
    <span class="sub">가로선 = ±1 표준편차</span></div>
  {ladder_chart()}
  <p class="fignote">* 축 정의 수정 = 아래 이슈 ②를 논문 Eq.6-7 대로 고친 판. +0.90 회복되지만 −1.99 가 남습니다.</p>

  <h3>확인이 필요한 것 3가지</h3>
  <div class="tablewrap"><table>
    <thead><tr><th style="width:26%">이슈</th><th>내용</th><th style="width:20%">영향</th></tr></thead>
    <tbody>
      <tr><td><b>① <code>main_v2.py</code> 부재</b><br><span class="pill crit">전제 조건</span></td>
        <td>공식 스크립트 <code>scripts/icbhi_dual_patchmix_gaussian.sh:13</code> 이 <code>python main_v2.py</code> 를 부르는데,
        저장소에는 <code>main.py</code>·<code>ast_attention_map.py</code>·<code>dass_attention_map.py</code> 만 있습니다.
        <b>우리는 <code>main.py</code> 로 이식했습니다.</b></td>
        <td>논문 수치를 낸 코드와 다른 코드를 재현한 셈 — seed·환경으로 닫을 수 없음</td></tr>
      <tr><td><b>② patch-mix 축 정의</b><br><span class="pill warn">+0.90 회복</span></td>
        <td><code>vmamba_patchmix.py:1207</code> 에서 <code>norm_layer="LN2D"</code> → <code>channel_first=True</code> 가 되어
        patch_embed 출력이 <code>(B,96,32,256)</code> channel-first 인데, patch-mix 함수는 이를 <code>B,H,W,C</code> 로 받습니다.
        그 결과 <code>time_only</code> 가 주파수(32)를, <code>freq_only</code> 가 채널(96)을 섞고
        <b>시간축(256)은 한 번도 섞이지 않습니다.</b></td>
        <td>논문 Eq.6-7 대로 고치면 61.59 → 62.49</td></tr>
      <tr><td><b>③ <code>patch_mix_dual_axis</code> 는 죽은 코드</b><br><span class="pill warn">검증 중</span></td>
        <td><code>vmamba_patchmix.py:1524</code> 의 순차 mix 함수가 forward 에서 <code>&#39;&#39;&#39;&#8230;&#39;&#39;&#39;</code> 로 주석 처리돼 있고
        <code>else: raise</code> 뒤라 도달 불가입니다. 실제 dual 은 <code>main.py:338-352</code> 의 독립 forward 2회입니다.
        <b><code>main_v2.py</code> 가 이 죽은 코드를 되살렸나요?</b></td>
        <td>가정하고 10 run 검증 중 — 첫 seed 는 59.33 으로 오히려 낮음</td></tr>
    </tbody>
  </table></div>

  <div class="ask"><b>요청</b>
    <p><b><code>main_v2.py</code> 를 받을 수 있을까요?</b> <code>main.py</code> 와의 diff 만으로도 충분합니다.
    이게 없으면 seed·환경·하이퍼파라미터 어느 것으로도 −2.89 를 닫을 수 없습니다.</p>
    <p>여유가 되시면 — 위 ②의 축 정의는 논문 Eq.6-7 쪽이 맞나요? 그리고 ③의 죽은 코드를 <code>main_v2</code> 가 되살렸나요?</p>
  </div>

  <h2><span class="n">02</span>seed 분포 — 논문값이 우리 분포의 어디에 있나</h2>
  <p>세 편의 성격이 여기서 갈립니다. 주황 점 5개가 우리 seed 1–5, 파란 선이 논문 평균(옅은 띠는 논문 ±1σ)입니다.</p>
  {"".join(seed_chart(t,s,pm,ps,note,i) for i,(t,s,pm,ps,note) in enumerate(SEEDS))}
  <p><b>BTS-CARD 와 QLung 은 논문값이 사실상 우리 분포 안에 있고, Lung-SRAD 만 분포 바깥입니다.</b>
  이 차이 때문에 세 편을 한 가지 원인으로 묶을 수 없다고 판단했습니다.</p>

  <h2><span class="n">03</span>QLung — seed 는 원인이 아닙니다</h2>
  <p>공식 스크립트 <code>scripts/train_qlung_icbhi.sh:2</code> 가 <code>SEED="1 2 3 4 5"</code> 로
  <b>우리와 완전히 같은 seed</b> 입니다. 같은 seed 로 60.81 vs 62.01 이므로 seed·학습 순서 가설은 여기서 성립하지 않습니다.</p>

  <h3>AQS(음향 품질 점수) 해석 3종 — 어느 쪽도 논문에 못 미칩니다</h3>
  <p><code>util/get_acoustic_features.py</code> 의 정의는
  <code>compute_quality_score_entropy_rms(entropy_norm, rms, …)</code> 로 <b>첫 인자가 entropy</b> 인데,
  호출은 <code>util/icbhi_dataset.py:177</code> 에서 <code>(rms_energy, rms_energy, …)</code> 로
  <b>entropy 자리에 rms 를 중복 전달</b>합니다. 윗줄에서 계산한 <code>spectral_entropy</code> 는 쓰이지 않습니다.
  그러면 <code>raw = −0.7·rms + 0.3·rms_score</code> 가 되어 두 항이 상쇄되고 <b>AQS 가 사실상 상수</b>(0.639±0.049)가 됩니다.
  entropy 를 제대로 넣으면 0.466±0.105 로 <b>논문 Fig.1 분포와 일치</b>합니다.</p>
  {aqs_chart()}
  <p style="margin-top:16px"><b>③ 이 특히 중요합니다.</b> 논문 식 (5) 를 문자 그대로 계산하면
  <b>디지털 무음 사이클 71건의 AQS 가 <code>nan</code></b> 이 되고(전 대역 전력이 0이라 확률분포를 만들 수 없습니다),
  그 <code>nan</code> 이 손실로 전파돼 학습이 무너집니다 — AST 기준선(59.80)보다도 낮습니다.
  즉 <b>논문 식 (5) 는 ICBHI 에 문자 그대로 적용할 수 없습니다.</b></p>

  <div class="ask"><b>요청</b>
    <p><b>논문 수치를 낸 체크포인트를 하나만 주실 수 있을까요?</b> 그리고 <b><code>pip freeze</code></b> 도 부탁드립니다.
    (README 는 Python 3.10 / torch 2.0.1 인데 우리 툴킷은 3.11.5 / 2.7.0 이라, 지금 저자 스택으로 별도 venv 를 만들어
    공개 코드를 그대로 5 seed 돌리는 A/B 를 진행 중입니다.)</p>
    <p>여유가 되시면 — 위 인자 위치가 의도인가요? 그리고 <b>무음 사이클의 <code>nan</code> 엔트로피는 어떻게 처리하셨나요?</b></p>
  </div>

  <h2><span class="n">04</span>BTS-CARD — 공개 스크립트에는 seed 가 하나뿐입니다</h2>
  <p>공식 스크립트 <code>icbhi_causal_adversarial_debiasing_bts_meta_all.sh:7</code> 이 <code>SEED="111"</code> —
  <b>단일 seed</b> 입니다. 논문은 5 run 평균인데 그 5개가 스크립트에 없고, <b>우리는 111 을 한 번도 돌린 적이 없습니다</b>(지금 큐에 넣었습니다).</p>
  <p>스크립트와 논문의 학습 설정도 다릅니다 — 스크립트 <code>--epochs 15 --optimizer adam</code>,
  논문 §3.1 은 <b>30 epoch, AdamW</b>. 우리는 코드 우선 원칙으로 15ep/Adam 을 주 결과로 삼았고, 논문 설정 A/B 도 돌리는 중입니다.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>설정</th><th class="num">seed 1</th><th class="num">seed 2</th><th class="num">seed 3</th><th class="num">5 seed 평균</th></tr></thead>
    <tbody>
      <tr><td>스크립트 (15 ep / Adam) — 주 결과</td><td class="num">63.92</td><td class="num">63.57</td><td class="num">64.63</td><td class="num"><b>63.47±0.99</b></td></tr>
      <tr><td>논문 (30 ep / AdamW)</td><td class="num">63.55</td><td class="num">63.41</td><td class="num">64.53</td><td class="num">진행 중 (3/5)</td></tr>
    </tbody>
  </table></div>
  <p>seed 별로 짝지어 보면 63.92↔63.55, 63.57↔63.41, 64.63↔64.53 으로 <b>두 설정이 사실상 같습니다.</b>
  −1.16 을 설명하지 못합니다. 우리 5 run 의 best 달성 epoch 도 7·15·8·7·9 로 4/5 가 절반 지점에서 최고점을 찍어,
  epoch 을 늘려도 상승 여지가 크지 않습니다. <b>남는 것은 seed 집합입니다.</b></p>

  <div class="ask"><b>요청</b>
    <p><b>논문 5-run 평균의 실제 seed 5개가 무엇인가요?</b> 그대로 돌리면 1:1 비교가 끝납니다.</p>
    <p>여유가 되시면 — 논문 표의 학습 설정은 15 epoch/Adam(스크립트)인가요, 30 epoch/AdamW(본문 §3.1)인가요?</p>
  </div>

  <h2><span class="n">05</span>참고 — 지금까지 검증한 모델 전체</h2>
  <p>같은 파이프라인·같은 공식 스플릿·같은 지표로 재현한 <b>29편 전부</b>입니다.
  <b>Δ 는 우리 Score − 논문 Score</b>. 팀 논문 3편은 강조 표시했습니다.
  기준선류(SFT·전이학습)가 거의 전부 재현된다는 점이, 앞서 말씀드린 "파이프라인은 정상" 의 근거입니다.</p>
  {all_models_table()}
  <p class="fignote">Se/Sp/Score 순 표기 · 우리 값은 seed 1–5 평균±표준편차 ·
  <span class="pill ok">✅</span> 논문 ± 안 또는 −0.5 이내 · <span class="pill warn">⚠️</span> 미달 · <span class="pill crit">⚠️⚠️</span> −5 이상 큰 미달 ·
  ICBHI 열의 <code>—</code> 는 해당 없음입니다.</p>

  <h3>SPRSound 열을 읽는 법</h3>
  <p>대부분의 논문은 SPRSound 를 쓰지 않아 <b>비교할 목표치가 없습니다</b>. 그래도 같은 방법이 다른 데이터에서 어떻게 도는지 보려고
  각 논문의 ICBHI 레시피 그대로 SPRSound 표준 태스크(2022 inter-patient, 7-class)로 돌리고 있습니다 — <b>관찰이지 재현 판정이 아닙니다.</b>
  열에 네 가지 상태가 섞여 있어 구분해 적었습니다:</p>
  <div class="tablewrap"><table>
    <thead><tr><th style="width:20%">표기</th><th>뜻</th><th style="width:34%">해당 논문</th></tr></thead>
    <tbody>
      <tr><td class="num"><b>값 ± 편차</b></td><td>우리가 5 seed 완주한 관찰값(논문 목표치 없음)</td>
        <td>AST SFT 87.19 · Patch-Mix CL 88.35 · ADD-RSC 86.17 · LungAdapter 87.86<br>
        <span class="sub">논문이 SPRSound 를 쓴 6편(SCL-meta·M2D·OPERA·QLung·BTS-CARD·LungMix)은 목표치가 있어 Δ 가 붙습니다</span></td></tr>
      <tr><td class="num">학습 중 (n/5)</td><td>3차 배치, 지금 GPU 에서 도는 중</td><td>PAFA(1/5) · BTS · Audio-CLAP · LoRA-RSC · AMS-D · Prototype</td></tr>
      <tr><td class="num">제외</td><td>방법 구조상 성립하지 않음</td>
        <td>RespireNet·SG-SCL(SPRSound 는 <b>단일 청진기</b>라 장치 도메인 학습 불가) · PC-MCL(3-시그모이드 헤드가 7-class 표현 불가) · Gap-Aug·DDE-MAE(ICBHI 에서 이미 붕괴)</td></tr>
      <tr><td class="num">미착수</td><td>가능하지만 아직 큐에 안 넣음</td><td>SFT 기준선 5종 · MVST · <b>Lung-SRAD</b> · Domain Transfer</td></tr>
    </tbody>
  </table></div>
  <p class="fignote">SPRSound 점수가 85–89 로 ICBHI(50–65)보다 높은 것은 방법이 더 잘 통해서가 아니라,
  inter-test 에 Rhonchi·Stridor 가 0건이고 Normal 이 73% 라 <b>평가상 실질이 3-class</b> 이기 때문입니다 —
  <b>두 데이터의 절대값을 직접 비교하면 안 됩니다.</b></p>

  <h2><span class="n">06</span>요청 정리</h2>
  <ol class="checks">
    <li><b>Lung-SRAD — <code>main_v2.py</code></b><span>스크립트가 부르는 파일이 저장소에 없습니다. diff 만으로도 충분합니다. <b>이것 없이는 다른 작업이 무의미합니다.</b></span></li>
    <li><b>QLung — 체크포인트 1개 + <code>pip freeze</code></b><span>BTS-CARD 에서 이미 통한 방법입니다. 하루 안에 “학습만의 문제인지”가 판정됩니다.</span></li>
    <li><b>BTS-CARD — 논문 5-run 의 seed 목록</b><span>스크립트에는 111 하나뿐입니다. seed 만 알면 그대로 재현해 비교할 수 있습니다.</span></li>
    <li><b>세 편 공통 — <code>pip freeze</code></b><span>QLung README 는 torch 2.0.1, Lung-SRAD <code>requirements.txt</code> 는 timm 0.4.12·transformers 4.35.2 인데 우리는 각각 2.7.0 / 1.0.28 / 5.15.1 입니다.</span></li>
  </ol>

  <h2><span class="n">07</span>병행 실험 진행 현황 <span class="sub" style="font-size:14px;font-weight:400">— 2026-08-31 18:06</span></h2>
  <p>저자 답변 없이도 확인할 수 있는 것은 지금 GPU 에서 돌리고 있습니다. <b>중간 결과부터 적습니다</b> —
  완주 전 값은 그렇게 표시했으니 참고만 해주세요.</p>
  <div class="tablewrap"><table>
    <thead><tr><th style="width:30%">실험</th><th style="width:11%">진행</th><th style="width:24%">지금까지</th><th>읽는 법</th></tr></thead>
    <tbody>
      <tr><td><b>QLung 저자 환경 재현</b><br><span class="sub">py3.10 · torch 2.0.1+cu118 · timm 0.4.5, 저자 코드 무수정</span></td>
        <td class="num">2/5<br><span class="sub">epoch ~32/50</span></td>
        <td class="num">seed 1 <b>60.54</b><br>seed 2 <b>63.78</b><br><span class="sub">둘 다 학습 중 best</span></td>
        <td><b>가장 주목하는 실험입니다.</b> 우리 툴킷의 같은 seed 는 60.43 / 61.15 인데,
        저자 환경 seed 2 가 <b>+2.63 높고 이미 논문 62.01 을 넘었습니다</b>. 사실이라면 원인은 코드가 아니라 <b>환경</b>입니다.
        다만 <b>2 seed·중간값</b>이라 단정하기 이릅니다 — 5 seed 완주 후 확정하겠습니다.</td></tr>
      <tr><td><b>Lung-SRAD 죽은 코드 경로</b><br><span class="sub"><code>cl_mode=dual_seq</code> × 축 2종</span></td>
        <td class="num">1/10</td><td class="num">seed 1 <b>59.33</b></td>
        <td>현행 dual(61.59)보다 <b>2.3점 더 낮습니다</b>. 첫 신호는 “<code>main_v2</code> 가 이 죽은 코드를 되살린 것은 아니다” 쪽입니다.</td></tr>
      <tr><td><b>BTS-CARD 논문 설정</b><br><span class="sub">30 epoch / AdamW</span></td>
        <td class="num">4/5</td><td class="num">63.55 · 63.41<br>64.53 · 63.98<br><span class="sub">평균 63.87</span></td>
        <td>스크립트 설정 같은 seed 는 63.92 · 63.57 · 64.63 · 63.24(평균 63.84) — <b>사실상 동일</b>합니다.
        학습 설정 차이는 −1.16 의 원인이 아닙니다.</td></tr>
      <tr><td><b>BTS-CARD seed 111</b><br><span class="sub">공개 스크립트의 유일한 seed</span></td>
        <td class="num">0/2</td><td class="num">대기 중</td><td>스크립트와 정확히 같은 조건. seed 집합 가설의 직접 확인입니다.</td></tr>
      <tr><td><b>QLung 논문 식 (5) 문자 그대로</b></td>
        <td class="num">3/5</td><td class="num">54.68 · 61.00 · 54.36<br><span class="sub">평균 56.68</span></td>
        <td>seed 마다 6점씩 흔들립니다 — <code>nan</code> AQS 가 학습을 불안정하게 만든 형태입니다.</td></tr>
    </tbody>
  </table></div>

  <div class="card" style="margin-top:26px">
    <h3 style="margin-top:0">참고 — 툴킷 전체 학습 현황</h3>
    <p>누적 <b>407 run</b> 완주(실패 2건은 08-24 의 고아 프로세스). 지금 <b>91 run</b> 이 큐에 있고 GPU 8장이 모두 돌고 있습니다.
    이 중 팀 논문 진단이 <b>21 run</b>, 나머지는 SPRSound 확장(3·4차 배치 78 run)입니다.
    Lung-SRAD 의 SPRSound 확장도 방금 큐에 넣었습니다(FT 기준선 + dual 각 5 seed).</p>
    <p style="margin:14px 0 0" class="fignote">셋업 중 확인 — QLung 은 README 만으로 환경이 재현되지 않습니다:
    <code>models/ast.py:47</code> 이 <code>assert timm.__version__ == '0.4.5'</code> 로 timm 을 하드 고정하는데 README 에 없고,
    <code>librosa</code>·<code>scipy</code>·<code>soundfile</code> 이 설치 안내에서 빠져 있으며,
    torch 2.0.1 은 numpy 1.x 전용이라 numpy 2 를 깔면 <code>RuntimeError: Numpy is not available</code> 로 죽습니다.</p>
  </div>
</div>
''', encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
