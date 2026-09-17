"""
Lungmix official-code parity goldens — imports the official lungmix() (outputs/reference/lungmix/lungmix.py, Tigerdwgth/ICASSP-2025-Lungmix) as-is,
runs it with fixed seeds and saves wav/label/lam/mask/max_length. tests/unit/models/test_ast_lungmix.py checks that the toolkit port
(augmentations/lungmix.py) is bit-identical.

  .venv/bin/python scripts/reproduction/dump_lungmix_goldens.py --ref outputs/reference/lungmix --out tests/golden/lungmix

Cases: synthetic waveforms (pairs of different length, same length, smooth on/off) + (if available) a pair of 2 ICBHI official train cycles (9 s repeat, 16 kHz).
Each case is called once right after np.random.seed(s); torch.manual_seed(s) (depends on the global RNG streams).

Thread pinning: the last bit of the torch.mean(wav2) constant padding value depends on the number of CPU threads (reduction order on long waveforms; in the icbhi case
64/16 threads vs 1/4 threads differ by 1.5e-10). Both the dump and the tests pin torch.set_num_threads(GOLDEN_THREADS=1).
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

GOLDEN_THREADS = 1


def main():
    torch.set_num_threads(GOLDEN_THREADS)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="outputs/reference/lungmix")
    ap.add_argument("--out", default="tests/golden/lungmix")
    ap.add_argument("--data", default=".datasets/icbhi")
    args = ap.parse_args()
    ref = pathlib.Path(args.ref).resolve()
    sys.path.insert(0, str(ref))
    import lungmix as official  # noqa: E402  official file as-is

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "-C", str(ref), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    g = torch.Generator().manual_seed(1234)
    cases = []
    synth = [
        ("synth_diff_len", torch.randn(1, 4000, generator=g), torch.randn(1, 6500, generator=g) * 0.3, "normal", "crackle", False, 11),
        ("synth_same_len", torch.randn(1, 5000, generator=g) * 0.1, torch.randn(1, 5000, generator=g), "crackle", "wheeze", False, 22),
        ("synth_smooth", torch.randn(1, 3000, generator=g), torch.randn(1, 4500, generator=g), "wheeze", "wheeze", True, 33),
        ("synth_both_normal", torch.randn(1, 2500, generator=g), torch.randn(1, 2000, generator=g), "both", "normal", False, 44),
    ]
    for name, w1, w2, l1, l2, smooth, seed in synth:
        cases.append(dict(name=name, wav1=w1, wav2=w2, label1=l1, label2=l2, smooth=smooth, seed=seed))

    data = pathlib.Path(args.data)
    if (data / "index/index_cycle.csv").exists():
        from rsc_toolkit.datasets.index_dataset import IndexDataset
        from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

        pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=9.0, pad_type="repeat"))
        ds = IndexDataset(index_csv=data / "index/index_cycle.csv", data_root=data, sampling_rate=16000, sample_level="cycle",
                          split="train", split_file="rsc_toolkit/datasets/icbhi/data/official_split.txt", audio_preprocessor=pre)
        names = ["normal", "crackle", "wheeze", "both"]
        a, b = ds[0], ds[1]
        cases.append(dict(name="icbhi_pair_9s", wav1=a["audio"].reshape(1, -1).float(), wav2=b["audio"].reshape(1, -1).float(),
                          label1=names[int(a["label"])], label2=names[int(b["label"])], smooth=False, seed=55,
                          sample_ids=[a["sample_id"], b["sample_id"]]))

    results = []
    for c in cases:
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        wav, label, lam, l1, l2, w1p, w2p, mask, max_length = official.lungmix(
            c["wav1"].clone(), c["wav2"].clone(), c["label1"], c["label2"], smooth=c["smooth"])
        results.append(dict(name=c["name"], seed=c["seed"], smooth=c["smooth"], wav1=c["wav1"], wav2=c["wav2"],
                            label1=c["label1"], label2=c["label2"], wav=wav, label=label, lam=float(lam),
                            wav1_proc=w1p, wav2_proc=w2p, mask=mask, max_length=int(max_length),
                            sample_ids=c.get("sample_ids")))
        print(f"{c['name']}: out_len={wav.shape[1]} lam={lam:.4f} label={label} mask_vals={sorted(set(mask.flatten().tolist()))[:5]}")

    torch.save({"cases": results, "official_file": str(ref / "lungmix.py"), "official_commit": commit,
                "torch": torch.__version__, "numpy": np.__version__, "num_threads": GOLDEN_THREADS}, out / "lungmix_golden.pt")
    json.dump({"official_commit": commit, "official_file": str(ref / "lungmix.py"), "torch": torch.__version__,
               "numpy": np.__version__, "num_threads": GOLDEN_THREADS,
               "note": "wav1/wav2 of icbhi_pair_9s contain the original ICBHI official train cycle waveforms (public data, CC BY 4.0 family; 6.3 MB) as-is",
               "cases": [{k: r[k] for k in ("name", "seed", "smooth", "label", "lam", "max_length")}
                                                  for r in results]}, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
    print("saved", out / "lungmix_golden.pt")


if __name__ == "__main__":
    main()
