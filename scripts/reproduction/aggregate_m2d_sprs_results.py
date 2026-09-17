"""
M2D SPRSound (2022 inter-patient, 7-class) fine-tuning 5-seed reproduction check aggregation: outputs/m2d_sprs/<run>/best.json (+ outputs/m2d/logs/<run>.done EXIT=0 —
these runs are appended to the m2d launcher queue, so the completion markers live in the m2d launcher log directory).

  .venv/bin/python scripts/reproduction/aggregate_m2d_sprs_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # M2D (16x4, 230529) SPRS fine-tuning Score cited in EMBC'25 (2504.18004) Table II; weights for TASLP M2D-X SPRS 89.77 are not public
    "sprs-m2d-base": ("M2D/0.7 (16x4, public ckpt 230529) SPRS fine-tuning", 88.78, 0.43),
    # M2D-Resp (AS+Resp400K, 16x16): the paper (2505.15307) reports only the 7 OPERA tasks → no SPRS paper-reported value (reference only)
    "sprs-m2dresp":  ("M2D-Resp (AS+Resp400K, 16x16, public ckpt 250826) SPRS fine-tuning — no paper-reported value", None, None),
}
LOGS = pathlib.Path("outputs/m2d/logs")


def _completed(run):
    done = LOGS / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/m2d_sprs")
    groups = {k: [] for k in PAPER}
    partial = []
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run)
        base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups:
            if not _completed(run):
                partial.append(run)
                continue
            d = json.loads(best.read_text())
            groups[base].append((seed, d["S_p"], d["S_e"], d["icbhi_score"], d.get("epoch")))
    if partial:
        print("(incomplete runs — excluded from aggregation: " + ", ".join(partial) + ")")
    for base, rows in groups.items():
        if not rows:
            continue
        name, psc, pstd = PAPER[base]
        rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, sp, se, sc, ep in rows:
            print(f"  seed {seed}: Sp {sp:6.2f}  Se {se:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            sp, se, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            if psc is None:
                print("  no paper-reported value (reference only; compare: M2D/0.7 base SPRS 88.78±0.43)")
            else:
                print(f"  paper     : Score {psc:6.2f}±{pstd:.2f} (EMBC'25 Table II)   diff: {sc[0]-psc:+.2f}")


if __name__ == "__main__":
    main()
