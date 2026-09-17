"""
SG-SCL features[0]/[1] experiment aggregation (GitHub main.py:373 block).

Groups:
  icbhi-ast-sgscl          (baseline)  cls=feat0, con=feat0  — behaviour of clone 6656460
  icbhi-ast-sgscl-githead              cls=feat0, con=feat1  — GitHub HEAD 2ed1bac (latest official code)
  icbhi-ast-sgscl-clsfeat1             cls=feat1, con=feat1  — features[1] also for the output line (experiment)
Comparison: clsfeat1 − githead is the effect of the output line; githead − baseline is the effect of upstream commit 2ed1bac.
"""
import json, pathlib, statistics

GROUPS = {
    "icbhi-ast-sgscl":          ("baseline: cls feat0 · con feat0 (clone 6656460)", "outputs/sgscl"),
    "icbhi-ast-sgscl-githead":  ("GitHub HEAD 2ed1bac: cls feat0 · con feat1(GRL)", "outputs/sgscl"),
    "icbhi-ast-sgscl-clsfeat1": ("experiment: cls feat1(GRL) · con feat1(GRL)", "outputs/sgscl"),
}
DONE_DIRS = {"icbhi-ast-sgscl": "outputs/sgscl/logs",
             "icbhi-ast-sgscl-githead": "outputs/sgsclgithead/logs",
             "icbhi-ast-sgscl-clsfeat1": "outputs/sgsclclsfeat1/logs"}
PAPER = (79.87, 43.55, 61.71, 1.61)

def done_ok(base, run):
    p = pathlib.Path(DONE_DIRS[base]) / f"{run}.done"
    return p.exists() and "EXIT=0" in p.read_text()

def main():
    print("SG-SCL features[0]/[1] experiment (paper SG-SCL(AST): Sp 79.87 Se 43.55 Score 61.71±1.61)")
    partial = []
    for base, (name, root) in GROUPS.items():
        rows = []
        for d in sorted(pathlib.Path(root).glob(f"{base}-s[0-9]*")):
            run = d.name
            if run.replace(base + "-s", "").isdigit() is False:
                continue
            if any(run.startswith(other + "-") and other != base for other in GROUPS if len(other) > len(base)):
                continue
            bj = d / "best.json"
            if not bj.exists():
                continue
            if not done_ok(base, run):
                partial.append(run); continue
            j = json.loads(bj.read_text())
            rows.append((int(run.rsplit("-s", 1)[1]), j["S_p"], j["S_e"], j["icbhi_score"], j.get("epoch")))
        rows.sort()
        if not rows:
            continue
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for sd, sp, se, sc, ep in rows:
            print(f"  seed {sd}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}   vs paper {PAPER[2]:.2f}±{PAPER[3]:.2f}: {sc[0]-PAPER[2]:+.2f}")
    if partial:
        print("\n(incomplete runs — excluded from aggregation: " + ", ".join(sorted(partial)) + ")")

if __name__ == "__main__":
    main()
