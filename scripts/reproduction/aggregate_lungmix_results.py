"""
Lungmix (ICASSP 2025, Ge et al.) 5-seed reproduction check aggregation — outputs/lungmix/<run>/best.json (+ logs/<run>.done EXIT=0).

**Paper-spec re-implementation / assumption runs** (the official code only provides the augmentation function; epochs/wd/λ2/mixing policy/band-pass/padding etc. are assumptions — see the config comments).
**COMB-based model selection is not possible**: the ICBHI/SPR numbers in paper Table I come from the best checkpoint on COMB (ICBHI+SPR+HF test), and HF is out of scope.
Toolkit values are official-split best-on-test (no Se gate), so (a) Fig.1 AST SFT (61.35 / 83.67; selection criterion not reported — but the Fig.1 HF(Test) column
has the same three values (58.24/62.19/75.34) as the Table I 'w/o Mixup' HF row, so Fig.1 likely uses the same run/COMB selection as Table I) is the baseline paper-reported value,
(b) Table I is used only for relative comparisons 'augmentation − w/o Mixup'. SPRSound uses the 7→4 merged 4-class view (non-standard); 2022 inter-patient test assumed.

Row mapping (unconfirmed by the authors): apply_loudness_mask=true (Eq. 3 blending) run = Table I 'Lungmix (non-linear)';
apply_loudness_mask=false (official function output = random mask only) run = 'Lungmix (w/o loudness)' (§III-C "using only a random mask").
Mixing position: main runs (*-nonlinear/-none/-woloudness) = mix_stage=raw (mix the original segments, then 9 s; based on the maxlen(x_i,x_j) notation in Eq. 3, assumption);
*-padded = alternative that mixes after 9 s padding (optional run).

  .venv/bin/python scripts/reproduction/aggregate_lungmix_results.py
"""
import json
import pathlib
import re
import statistics

# run prefix → (display name, Table I value ICBHI→ICBHI / SPR→SPR, note)
PAPER_TABLE1 = {
    "icbhi-lungmix-none":      ("AST w/o Mixup",                52.88, None),
    "icbhi-lungmix-nonlinear": ("Lungmix (non-linear)",         53.35, "apply_loudness_mask=true: Eq. 3 blending, mix_stage=raw (unconfirmed by the authors)"),
    "icbhi-lungmix-woloudness": ("Lungmix (w/o loudness)",      58.53, "apply_loudness_mask=false: official function output (random mask only)"),
    "icbhi-lungmix-padded":    ("Lungmix (non-linear, padded mixing)", 53.35, "mix_stage=padded alternative (mix after 9 s padding; ~25% constant padding expected in the output)"),
    "icbhi-lungmix-overlap1":  ("Lungmix (non-linear, overlap=one)", 53.35, "Eq. 8 OR interpretation (overlap=1) alternative"),
    "icbhi-lungmix-linear":    ("Lungmix (linear)",             52.13, None),
    "icbhi-lungmix-combined":  ("Lungmix (combined)",           53.78, "λ2 = L_CE/L_mix assumed"),
    "icbhi-lungmix-vanilla":   ("vanilla Mixup",                58.53, "waveform mixup, α=0.5 assumed"),
    "icbhi-lungmix-patchmix":  ("Lungmix + Patch-mix",          52.82, None),
    "sprs-lungmix-none":       ("AST w/o Mixup",                80.44, None),
    "sprs-lungmix-nonlinear":  ("Lungmix (non-linear)",         80.61, "apply_loudness_mask=true: Eq. 3 blending, mix_stage=raw (unconfirmed by the authors)"),
    "sprs-lungmix-woloudness": ("Lungmix (w/o loudness)",       76.27, "apply_loudness_mask=false: official function output (random mask only)"),
    "sprs-lungmix-padded":     ("Lungmix (non-linear, padded mixing)", 80.61, "mix_stage=padded alternative"),
    "sprs-lungmix-overlap1":   ("Lungmix (non-linear, overlap=one)", 80.61, "Eq. 8 OR interpretation (overlap=1) alternative"),
    "sprs-lungmix-linear":     ("Lungmix (linear)",             77.58, None),
    "sprs-lungmix-combined":   ("Lungmix (combined)",           80.21, None),
    "sprs-lungmix-vanilla":    ("vanilla Mixup",                53.85, "SPR/HF columns have identical values — possible typo in the paper"),
    "sprs-lungmix-patchmix":   ("Lungmix + Patch-mix",          79.06, None),
}
FIG1_ANCHOR = {"icbhi": 61.35, "sprs": 83.67}   # AST simple fine-tune (Fig.1; selection criterion not reported — HF column equals the Table I w/o Mixup HF row)
TABLE2_ICBHI_NONLINEAR = (79.08, 47.51)          # ICBHI→COMB Sp/Se (out of scope, reference only)


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
    print("Lungmix 5-seed reproduction check aggregation — paper-spec re-implementation / assumption runs / official-split best-on-test (COMB selection not possible) / SPRS 4-class merged (non-standard)")
    for ds in ("icbhi", "sprs"):
        print(f"\n########## {ds.upper()} → {ds.upper()}   (Fig.1 AST SFT paper-reported value {FIG1_ANCHOR[ds]:.2f}, selection criterion not reported; "
              f"Fig.1 HF column = Table I w/o Mixup HF row → likely the same COMB selection)")
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
            line = f"  ours mean : Sp {sp[0]:6.2f}±{sp[1]:.2f}  Se {se[0]:6.2f}±{se[1]:.2f}  Score {sc[0]:6.2f}±{sc[1]:.2f}"
            if key == base_key:
                line += f"   | vs Fig.1 reference {FIG1_ANCHOR[ds]:.2f}: {sc[0]-FIG1_ANCHOR[ds]:+.2f}"
            print(line)
            print(f"  Table I   : {paper:6.2f} (single value, best by COMB; not directly comparable)")
            if key != base_key and base_mean is not None:
                print(f"  relative  : ours (aug − w/o Mixup) {sc[0]-base_mean:+.2f}  vs  paper {paper-paper_base:+.2f}")
            if key == "icbhi-lungmix-nonlinear":
                print(f"  reference: Table II ICBHI→COMB non-linear Sp/Se {TABLE2_ICBHI_NONLINEAR[0]:.2f}/{TABLE2_ICBHI_NONLINEAR[1]:.2f} (out of scope)")


if __name__ == "__main__":
    main()
