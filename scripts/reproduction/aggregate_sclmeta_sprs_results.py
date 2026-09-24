"""
SCL-metadata SPRSound (2022 inter-patient, 7-class) 5-seed reproduction check aggregation: outputs/sclmeta_sprs/<run>/best.json (+ logs/<run>.done EXIT=0).
Order follows the paper: Se/Sp/Sc (HS is not computed because the code does not have it).

  .venv/bin/python scripts/reproduction/aggregate_sclmeta_sprs_results.py
"""
import json
import pathlib
import re
import statistics

PAPER = {  # arXiv 2210.16192 SPRS inter (Se / Sp / Sc); the paper uses 200ep, 10 runs; code default 400ep
    "sprs-sclmeta-ce":     ("CE",     76.89, 92.35, 84.62),
    "sprs-sclmeta-scl":    ("SCL",    80.69, 90.49, 85.59),
    "sprs-sclmeta-mscl":   ("M-SCL",  82.24, 88.62, 85.43),
    "sprs-sclmeta-hybrid": ("Hybrid", None, None, None),
}


def _completed(root, run):
    done = root / "logs" / f"{run}.done"
    return done.exists() and "EXIT=0" in done.read_text()


def main():
    root = pathlib.Path("outputs/sclmeta_sprs")
    groups = {k: [] for k in PAPER}
    for best in sorted(root.glob("*/best.json")):
        run = best.parent.name
        m = re.fullmatch(r"(.+?)(?:-s(\d+))?", run); base, seed = m.group(1), int(m.group(2) or 1)
        if base in groups and _completed(root, run):
            d = json.loads(best.read_text()); groups[base].append((seed, d["S_e"], d["S_p"], d["icbhi_score"], d.get("epoch")))
    for base, rows in groups.items():
        if not rows:
            continue
        name, pse, psp, psc = PAPER[base]; rows.sort()
        print(f"\n== {name} ({base}) — {len(rows)} seeds ==")
        for seed, se, sp, sc, ep in rows:
            print(f"  seed {seed}: Se {se:6.2f}  Sp {sp:6.2f}  Score {sc:6.2f}  (epoch {ep})")
        if len(rows) >= 2:
            ms = lambda i: (statistics.mean([r[i] for r in rows]), statistics.stdev([r[i] for r in rows]))  # noqa: E731
            se, sp, sc = ms(1), ms(2), ms(3)
            print(f"  ours mean : Se {se[0]:6.2f}±{se[1]:.2f}  Sp {sp[0]:6.2f}±{sp[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}")
            if psc is not None:
                print(f"  paper     : Se {pse:6.2f}  Sp {psp:6.2f}  Score {psc:6.2f} (200ep, 10 runs)   Score diff: {sc[0]-psc:+.2f}")


if __name__ == "__main__":
    main()
