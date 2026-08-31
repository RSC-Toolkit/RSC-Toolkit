# -*- coding: utf-8 -*-
"""docs/RESULTS_BY_DATASET.md §1 요약표를 그대로 읽어 HTML 대장을 만든다 (수치 손으로 옮기지 않기)."""
import re, html, pathlib, datetime

SRC = pathlib.Path("docs/RESULTS_BY_DATASET.md")
OUT = pathlib.Path("/tmp/claude-10008/-raid-doyoung-projects-RSC-Toolkit/88feaa60-f05e-4f32-8234-95e341f584ce/scratchpad/rsc_repro_ledger.html")

lines = SRC.read_text().splitlines()

def table_after(header_sub, start=0, limit=None):
    """header_sub 를 포함하는 표 머리 다음의 데이터 행들을 반환."""
    for i in range(start, len(lines)):
        if lines[i].startswith("|") and header_sub in lines[i]:
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                rows.append(cells); j += 1
                if limit and len(rows) >= limit: break
            return rows, i
    return [], -1

main_rows, _ = table_after("| 모델 | 학습 방식 |")
gain_rows, _ = table_after("| 논문 | 이득(논문) |")

def md(t):
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"~~(.+?)~~", r"<s>\1</s>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t

def verdict_pill(v):
    core = v.replace("¹","").replace("²","").replace("³","").replace("⁵","").replace("⁶","").replace("⁷","")
    core = re.sub(r"[⁰-⁹\s]", "", core)
    if core == "✅✅": return '<span class="pill ok">✅✅ 기준선·확장 모두</span>'
    if core == "✅":   return '<span class="pill ok">✅ 재현</span>'
    if core == "⚠️⚠️": return '<span class="pill crit">⚠️⚠️ 큰 미달</span>'
    if core == "⚠️":   return '<span class="pill warn">⚠️ 미달</span>'
    if core == "⚠️/✅": return '<span class="pill warn">⚠️</span><span class="pill ok">✅</span>'
    if core == "참고":  return '<span class="pill na">참고</span>'
    if core == "제외":  return '<span class="pill na">제외</span>'
    return f'<span class="pill na">{md(v)}</span>'

def delta_bar(d):
    """Δ 를 발산형 막대로. ±3 을 눈금 끝으로 두고 그 밖은 잘린 표시."""
    m = re.search(r"([+−-]?\d+\.\d+)", d.replace("**",""))
    if not m: return f'<span class="dnum na">{md(d) or "—"}</span>'
    v = float(m.group(1).replace("−","-"))
    w = min(abs(v), 3.0) / 3.0 * 50.0
    cls = "pos" if v >= 0 else "neg"
    side = f'left:50%;width:{w:.1f}%' if v >= 0 else f'right:50%;width:{w:.1f}%'
    clipped = ' clipped' if abs(v) > 3.0 else ''
    return (f'<span class="dwrap"><span class="dtrack"><span class="dfill {cls}{clipped}" style="{side}"></span>'
            f'<span class="dzero"></span></span>'
            f'<span class="dnum {cls}">{v:+.2f}</span></span>')

body = []
for r in main_rows:
    if len(r) < 10: continue
    idx, model, how, i_our, i_pap, i_d, s_our, s_pap, s_d, verd = r[:10]
    excluded = "~~" in model
    body.append(f'''<tr{' class="dim"' if excluded else ''}>
      <td class="idx">{md(idx)}</td>
      <td class="model">{md(model)}<small>{md(how)}</small></td>
      <td class="num">{md(i_our) or "—"}</td>
      <td class="num pap">{md(i_pap) or "—"}</td>
      <td class="delta">{delta_bar(i_d)}</td>
      <td class="num">{md(s_our) or "—"}</td>
      <td class="num pap">{md(s_pap) or "—"}</td>
      <td class="delta">{delta_bar(s_d)}</td>
      <td class="verd">{verdict_pill(verd)}</td>
    </tr>''')

gain = []
for r in gain_rows:
    if len(r) < 4: continue
    name, pg, og, disc = r[:4]
    m = re.search(r"([+−-]?\d+\.\d+)", og.replace("**",""))
    v = float(m.group(1).replace("−","-")) if m else 0.0
    gain.append(f'''<tr><td class="model">{md(name)}</td><td class="num">{md(pg)}</td>
      <td class="num {'pos' if v>=0 else 'neg'}"><b>{md(og)}</b></td><td class="extra">{md(disc)}</td></tr>''')

ok  = sum(1 for r in main_rows if len(r)>9 and r[9].startswith("✅"))
bad = sum(1 for r in main_rows if len(r)>9 and r[9].startswith("⚠️"))
now = "2026-08-31 17:55 KST"

OUT.write_text(f'''<title>RSC-Toolkit 재현 대장</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700&family=Noto+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {{
    --bg:#F6F8F6; --surface:#FFFFFF; --surface-2:#EEF2EF; --line:#D6DDD8;
    --ink:#1A231F; --muted:#5B6862; --accent:#0E7169; --accent-ink:#0A5A54;
    --ok-bg:#DCEFE3; --ok-ink:#1F6B41;
    --warn-bg:#F8E9C9; --warn-ink:#8A5A10;
    --crit-bg:#F6DAD3; --crit-ink:#8E3218;
    --na-bg:#ECEDEA; --na-ink:#6C746F;
    --pos:#1F6B41; --neg:#A8452A; --track:#E3E8E4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#101613; --surface:#171F1B; --surface-2:#1E2823; --line:#2C3832;
      --ink:#E4EAE6; --muted:#98A59E; --accent:#3FB8A9; --accent-ink:#7FD4C9;
      --ok-bg:#1B3A2A; --ok-ink:#8FD9AD;
      --warn-bg:#3E2E10; --warn-ink:#F0C46A;
      --crit-bg:#41211A; --crit-ink:#F0A28A;
      --na-bg:#222824; --na-ink:#9AA39D;
      --pos:#6FCB98; --neg:#E58C6F; --track:#26302B;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#101613; --surface:#171F1B; --surface-2:#1E2823; --line:#2C3832;
    --ink:#E4EAE6; --muted:#98A59E; --accent:#3FB8A9; --accent-ink:#7FD4C9;
    --ok-bg:#1B3A2A; --ok-ink:#8FD9AD;
    --warn-bg:#3E2E10; --warn-ink:#F0C46A;
    --crit-bg:#41211A; --crit-ink:#F0A28A;
    --na-bg:#222824; --na-ink:#9AA39D;
    --pos:#6FCB98; --neg:#E58C6F; --track:#26302B;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background:var(--bg); color:var(--ink); font-family:"Noto Sans KR","Apple SD Gothic Neo",system-ui,sans-serif; font-size:14px; line-height:1.55; margin:0; }}
  .wrap {{ max-width:1520px; margin:0 auto; padding:40px 24px 72px; display:grid; gap:0; }}
  .eyebrow {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent-ink); }}
  h1 {{ font-family:"Noto Serif KR",serif; font-weight:700; font-size:34px; line-height:1.2; margin:8px 0 10px; text-wrap:balance; }}
  .lede {{ color:var(--muted); max-width:74ch; margin:0; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:12px; margin:26px 0 8px; }}
  .stat {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:13px 15px; display:grid; gap:1px; }}
  .stat .k {{ font-size:12px; color:var(--muted); }}
  .stat .v {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .stat .s {{ font-size:11.5px; color:var(--muted); }}
  h2 {{ font-family:"Noto Serif KR",serif; font-size:20px; font-weight:600; margin:42px 0 4px; }}
  .h2note {{ color:var(--muted); margin:0 0 14px; font-size:13px; max-width:82ch; }}
  .tablewrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:var(--surface); }}
  table {{ border-collapse:collapse; width:100%; min-width:1200px; }}
  th,td {{ text-align:left; vertical-align:top; padding:9px 11px; border-bottom:1px solid var(--line); }}
  thead th {{ position:sticky; top:0; background:var(--surface-2); font-size:11.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); font-weight:500; z-index:1; }}
  thead th.grp {{ color:var(--accent-ink); }}
  tbody tr:last-child td {{ border-bottom:0; }}
  tbody tr:hover td {{ background:color-mix(in srgb, var(--surface-2) 55%, transparent); }}
  tbody tr.dim td {{ opacity:.5; }}
  td.idx {{ font-family:"IBM Plex Mono",monospace; color:var(--muted); font-variant-numeric:tabular-nums; width:2.2em; }}
  td.model {{ font-weight:700; min-width:150px; }}
  td.model small {{ display:block; font-weight:400; color:var(--muted); font-size:11.5px; line-height:1.4; margin-top:2px; }}
  td.num {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; font-size:12.5px; white-space:nowrap; }}
  td.num.pap {{ color:var(--muted); }}
  td.extra {{ color:var(--muted); font-size:12.5px; min-width:150px; }}
  td.verd {{ white-space:nowrap; }}
  td.delta {{ width:132px; }}
  .dwrap {{ display:grid; gap:2px; }}
  .dtrack {{ position:relative; height:6px; background:var(--track); border-radius:3px; }}
  .dfill {{ position:absolute; top:0; bottom:0; border-radius:3px; }}
  .dfill.pos {{ background:var(--pos); }}
  .dfill.neg {{ background:var(--neg); }}
  .dfill.clipped {{ border-radius:3px 0 0 3px; }}
  .dzero {{ position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--line); }}
  .dnum {{ font-family:"IBM Plex Mono",monospace; font-size:12px; font-variant-numeric:tabular-nums; }}
  .dnum.pos {{ color:var(--pos); }} .dnum.neg {{ color:var(--neg); }} .dnum.na {{ color:var(--muted); }}
  td.num.pos {{ color:var(--pos); }} td.num.neg {{ color:var(--neg); }}
  .pill {{ display:inline-block; padding:1px 8px; border-radius:999px; font-size:11.5px; font-weight:500; white-space:nowrap; margin-right:3px; }}
  .ok {{ background:var(--ok-bg); color:var(--ok-ink); }}
  .warn {{ background:var(--warn-bg); color:var(--warn-ink); }}
  .crit {{ background:var(--crit-bg); color:var(--crit-ink); }}
  .na {{ background:var(--na-bg); color:var(--na-ink); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px 18px; }}
  .card h3 {{ font-family:"Noto Serif KR",serif; font-size:15px; margin:0 0 6px; }}
  .card p {{ margin:0; color:var(--muted); font-size:13px; }}
  .card p + p {{ margin-top:8px; }}
  .card b {{ color:var(--ink); }}
  code {{ font-family:"IBM Plex Mono",monospace; font-size:.92em; background:var(--surface-2); padding:1px 4px; border-radius:3px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:8px 18px; color:var(--muted); font-size:12.5px; margin-top:12px; }}
  a {{ color:var(--accent-ink); }}
  a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">RSC-Toolkit · ICBHI 2017 공식 60/40 · SPRSound 2022 inter-patient</div>
    <h1>호흡음 분류 논문 재현 대장</h1>
    <p class="lede">재현 대상 29편의 이식(P0)과 5-seed 재학습(P1) 현황. 값은 <b>Se / Sp / Score</b>, 우리 수치는 seed 1–5 평균±표준편차.
    Δ 는 우리 Score − 논문 Score 이고, 막대는 ±3.00 을 눈금 끝으로 둔다(그보다 크면 끝까지 채운다). 기준 시각 {now}.</p>
  </header>

  <div class="stats">
    <div class="stat"><span class="k">재현 대상</span><span class="v">29</span><span class="s">SSAST 배제 · Domain Transfer·Prototype 추가</span></div>
    <div class="stat"><span class="k">✅ 재현 성립</span><span class="v">{ok}</span><span class="s">논문 ± 안 또는 −0.5 이내</span></div>
    <div class="stat"><span class="k">⚠️ 미달</span><span class="v">{bad}</span><span class="s">대부분 이득만 사라지는 형태</span></div>
    <div class="stat"><span class="k">완주 run</span><span class="v">404</span><span class="s">누적 학습 종료</span></div>
    <div class="stat"><span class="k">진행 중</span><span class="v">50</span><span class="s">SPRS 3차 29 · 팀 진단 21</span></div>
  </div>

  <h2>요약 — 논문 29편</h2>
  <p class="h2note"><b>SPRSound 열 읽는 법</b> — 대부분의 논문은 이 데이터를 쓰지 않아 <b>비교할 목표치가 없다</b>.
  그래도 같은 방법이 다른 데이터에서 어떻게 도는지 보려고 각 논문의 ICBHI 레시피 그대로 돌리고 있다(관찰이지 판정이 아니다).
  <b>값 ¹⁴</b> = 우리 5 seed 완주 관찰값 · <b>학습 중 (n/5)</b> = 3차 배치 진행 중 ·
  <b>제외 ¹⁵</b> = 방법 구조상 성립 불가(단일 청진기·라벨 공간·ICBHI 에서 이미 붕괴) ·
  <b>미착수 ¹⁶</b> = 가능하지만 아직 큐에 안 넣음. ICBHI 열의 <code>—</code> 는 해당 없음.
  Δ 가 비어 있는 칸은 논문 목표치가 없거나 논문 내부 불일치로 절대값 비교가 성립하지 않는 경우다.
  SPRSound 절대값은 ICBHI 와 직접 비교하면 안 된다(inter-test 가 실질 3-class).</p>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>#</th><th>모델 · 학습 방식</th>
        <th class="grp">ICBHI 우리</th><th>ICBHI 논문</th><th>Δ</th>
        <th class="grp">SPRSound 우리</th><th>SPRSound 논문</th><th>Δ</th>
        <th>판정</th>
      </tr></thead>
      <tbody>
{chr(10).join(body)}
      </tbody>
    </table>
  </div>
  <div class="legend">
    <span><span class="pill ok">✅</span> 논문 ± 안 또는 −0.5 이내</span>
    <span><span class="pill warn">⚠️</span> 미달</span>
    <span><span class="pill crit">⚠️⚠️</span> −10 이상 또는 두 데이터 모두 미달</span>
    <span><span class="pill na">참고·제외</span> 판정 대상 아님</span>
  </div>

  <h2>가장 큰 발견 — 기준선은 맞고 이득만 사라진다</h2>
  <p class="h2note">이득 = 같은 논문 안에서 (방법 Score − 기준선 Score). 경계가 방법의 난이도가 아니라
  <b>학습 명세 공개 수준</b>에 그어진다. 코드가 완비된 논문은 이득의 방향까지 재현되고, 무언가 빠진 논문은 전부 음수로 뒤집힌다.</p>
  <div class="tablewrap">
    <table style="min-width:760px">
      <thead><tr><th>논문</th><th>이득(논문)</th><th>이득(우리)</th><th>학습 명세 공개 수준</th></tr></thead>
      <tbody>
{chr(10).join(gain)}
      </tbody>
    </table>
  </div>

  <h2>지금 돌고 있는 것</h2>
  <div class="cards">
    <div class="card"><h3>A/B 검증 10 run</h3>
      <p><b>BTS-CARD</b> — 공식 스크립트는 15 epoch·Adam 인데 논문 본문은 30 epoch·AdamW 다. P0 원칙대로 코드를 따랐으니, 논문 설정으로도 5 seed 돌려 격차 −1.16 의 원인인지 가린다.</p>
      <p><b>QLung</b> — 논문 식 5 를 문자 그대로 계산하면 디지털 무음 사이클 71 건이 <code>nan</code> AQS 가 된다. 인자 버그 수정판(60.38±0.46)은 이미 끝냈고, 남은 변형이 이것이다.</p></div>
    <div class="card"><h3>SPRSound 확장 3차 배치 30 run</h3>
      <p>PAFA · BTS · Audio-CLAP · LoRA-RSC · AMS-D · Prototype 각 5 seed. <b>각 논문의 ICBHI 레시피를 그대로</b> 쓰고, 클래스만 M2D 기준으로 7개 전부 사용한다.</p>
      <p><b>PC-MCL 은 제외</b> — 방법의 핵심인 3-시그모이드 다중라벨 헤드가 ICBHI 4-class = 2×2 속성 구조에 묶여 있어, Rhonchi·Stridor 를 포함한 7-class 로는 표현할 수 없다. 라벨 공간을 바꾸면 방법 자체가 달라진다.</p></div>
    <div class="card"><h3>SPRS 가중 CE 1:344 편중 — 결론</h3>
      <p>ICBHI 산식을 SPRSound 분포에 그대로 적용하면 Normal:Stridor 가중치가 <b>1:344</b> 가 된다. 15 run 을 돌린 결과 평균 손실은 −0.53 ~ −1.06 으로 작지만, <code>sprs-ast-sft-s4</code> <b>한 seed 가 Sp 81.35 / Se 90.75</b> 로 무너졌다.</p>
      <p>드문 seed 단위 사건이라 clip·정규화 같은 개입은 하지 않는다(논문에 없는 산식이라 P0 위반). <b>가중을 주 결과로, 비가중을 민감도 대조로 병기</b>한다.</p></div>
  </div>

  <h2>운영</h2>
  <div class="cards">
    <div class="card"><h3>GPU 배치</h3><p>GPU 당 1개·공유 금지, 8 GPU 사용, 시간당 1개 투입, 83°C 이상 회피.
    자동 스케줄러는 <b>워커가 0개인 번들만</b> 집으므로 이미 도는 번들에는 추가 투입하지 않는다 — 그래서 매시간 수동으로 1개씩 늘린다.</p></div>
    <div class="card"><h3>최근 고친 것</h3>
    <p><b>피처 캐시 체인 가드</b> — 캐시 키가 프로세서 설정만 담아, 같은 데이터셋을 다른 전처리로 캐시하면 조용히 남의 피처를 읽었다. 데이터셋별 체인을 기록하고 어긋나면 예외를 던진다(기존 캐시는 무효화하지 않는다). 실제 충돌 1건을 이 과정에서 잡았다.</p>
    <p><b>CLAP 특징 추출</b> — torch 1-D 텐서를 넘기면 transformers 가 3 TiB 배열을 만들려다 죽던 문제를 numpy 1-D 로 우회.</p></div>
    <div class="card"><h3>대기 중</h3>
    <p><b>저자 답변</b> — Lung-SRAD(<code>main_v2.py</code>·축 정의) · BTS-CARD(실제 epoch·optimizer) · QLung(AQS 변형) 외 6편.</p>
    <p><b>사용자 결정</b> — 팀 논문 저자 환경 A/B(venv 3개, ~1일) · PC-MCL 의 SPRSound 확장 여부.</p></div>
  </div>
</div>
''', encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes ·", len(main_rows), "rows ·", len(gain_rows), "gain rows")
