"""
PC-MCL (wa976/PC-MCL 85a4e22) official-code parity goldens — runs the official code with minimal patches (stubs, PYTHONPATH=models, forced CPU) and saves to tests/golden/pcmcl/.

  # (1) dataset sample sequence, cut_pad, fbank, SpecAugment, 3→4 conversion, loss  (toolkit venv: needs torchaudio, CUDA hidden)
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/reproduction/dump_pcmcl_goldens.py --part data
  # (2) BEATs features, 3-logit, PM-logit, loss (toolkit venv; official models/BEATs code + real iter3+ AS2M ckpt)
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/reproduction/dump_pcmcl_goldens.py --part beats
  # (3) AST features, 3-logit (timm 0.4.5 reference venv, CPU; uses the fbank_input.pt produced by (1))
  CUDA_VISIBLE_DEVICES= .venv-ref-timm045/bin/python scripts/reproduction/dump_pcmcl_goldens.py --part ast

The official data folder convention (args.data_folder/icbhi_dataset/{audio_test_data,official_split.txt}) is reproduced with a symlinked folder
(the original data .datasets/icbhi is only read; only the 920 wav/txt pairs are linked — auxiliary files such as filename_*.txt break the original get_annotations).

Goldens (dataset.pt):
- construction statistics of ConcatenationMultilabelDataset(train, seed 1, T=10 s, mixing_prob 1.0, PM on ssl_prob 0.3) (number of cycles, per-class/short-cycle counts,
  number of patients (=recordings)), all_cycles order (file name, label, length), label/ssl_label/waveform summaries (first 1 s, sum, abs sum; full waveform for some)
  of a fixed index sequence right after random.seed(1) (6 original, 30 concatenated, 30 PM), and the cut_pad waveforms of 4 test-set cycles.
- fbank_input.pt: generate_fbank(2 waveforms: original and concatenated) (2, 998, 128) — AST golden input and toolkit mel comparison
- specaug.pt: result of applying SpecAugment(icbhi_ast_sup, mean) to a batch with a fixed seed
- conversion.pt: the 8 3-label combinations → convert_original, get_score(hits, counts)
- beats.pt / ast.pt: features, logits, losses (random heads made deterministic with det_fill)
"""
import argparse
import importlib.machinery
import json
import os
import pathlib
import random
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE_IDX_ORIG = [0, 1, 2, 100, 2000, 4141]
N_MIXED_GOLD, N_SSL_GOLD = 30, 30
FULL_WAV_PER_GROUP = 3


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def make_data_folder(scratch: pathlib.Path, icbhi_dir: pathlib.Path, split_file: pathlib.Path) -> pathlib.Path:
    audio = scratch / "icbhi_dataset" / "audio_test_data"
    audio.mkdir(parents=True, exist_ok=True)
    n = 0
    for wav in sorted(icbhi_dir.glob("*.wav")):
        txt = icbhi_dir / (wav.stem + ".txt")
        if not txt.exists():
            continue
        for src in (wav, txt):
            dst = audio / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve())
        n += 1
    dst = scratch / "icbhi_dataset" / "official_split.txt"
    if not dst.exists():
        dst.symlink_to(split_file.resolve())
    print(f"data folder {scratch} ({n} recordings)")
    return scratch


def official_args(data_folder, seed=1):
    return types.SimpleNamespace(
        seed=seed, data_folder=str(data_folder), test_fold="official", class_split="lungsound",
        sample_rate=16000, desired_length=10, pad_types="repeat", n_mels=128, model="beats",
        mixing_prob=1.0, mixing_alpha=0.5, conditional_mixing=False, mixing_label_strategy="additive",
        enable_ssl=True, ssl_method="coherence", ssl_prob=0.3, ssl_separate_training=False,
        ablation_mode="default", threshold=0.5, specaug_policy="icbhi_ast_sup", specaug_mask="mean", nospec=False,
    )


def _wav_summary(w: torch.Tensor, full: bool):
    w = w.reshape(-1).float()
    d = {"len": int(w.numel()), "head": w[:16000].clone(), "sum": float(w.double().sum()), "abssum": float(w.double().abs().sum())}
    if full:
        d["wav"] = w.clone()
    return d


def part_data(args, out: pathlib.Path):
    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    _stub("tkinter", W="w")
    _stub("torchlibrosa"); _stub("torchlibrosa.stft", Spectrogram=object, LogmelFilterBank=object)
    _stub("nlpaug"); _stub("nlpaug.augmenter"); _stub("nlpaug.augmenter.audio")
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        _stub("matplotlib"); _stub("matplotlib.pyplot")
    scratch = make_data_folder(pathlib.Path(args.scratch), ROOT / ".datasets" / "icbhi",
                               ROOT / "rsc_toolkit/datasets/icbhi/data/official_split.txt")
    from util.dataset_pcmcl import ConcatenationMultilabelDataset  # noqa: E402
    from util.icbhi_util import generate_fbank, get_score  # noqa: E402
    from util.augmentation import SpecAugment  # noqa: E402
    from method.conversion import convert_original  # noqa: E402

    a = official_args(scratch, seed=args.seed)
    ds = ConcatenationMultilabelDataset(train_flag=True, transform=None, args=a, print_flag=True)
    n = len(ds.all_cycles)
    n_mixed = int(n * ds.mixing_prob)
    n_ssl = int(n * ds.ssl_prob)
    assert len(ds) == n + n_mixed + n_ssl
    meta = {
        "seed": args.seed, "n_cycles": n, "dataset_size": len(ds), "n_mixed": n_mixed, "n_ssl": n_ssl,
        "target_samples": ds.target_samples, "half_target_samples": ds.half_target_samples,
        "class_counts": [len(c) for c in ds.classwise_cycle_list],
        "short_counts": [len(c) for c in ds.short_classwise_cycle_list],
        "n_patients": len(ds.patient_cycles),
        "patient_ids": list(ds.patient_cycles.keys()),
        "patient_sizes": [len(v) for v in ds.patient_cycles.values()],
        "cycle_filenames": [c["filename"] for c in ds.all_cycles],
        "cycle_labels": [int(c["original_label"]) for c in ds.all_cycles],
        "cycle_lengths": [int(c["length"]) for c in ds.all_cycles],
        "combinations": [list(c) for c in ds.mixing_combinations],
    }
    raw_cycles = [ds.all_cycles[i]["audio"].clone() for i in range(6)]

    idx_list = list(SAMPLE_IDX_ORIG) + list(range(n, n + N_MIXED_GOLD)) + list(range(n + n_mixed, n + n_mixed + N_SSL_GOLD))
    random.seed(args.seed)                      # same python random stream start as the original (reference for comparing the sample sequence)
    samples = []
    for k, i in enumerate(idx_list):
        wav, label, is_mixed, orig, ssl_label, ssl_task = ds[i]
        group = "orig" if i < n else ("mixed" if i < n + n_mixed else "ssl")
        full = (group == "orig" and k < FULL_WAV_PER_GROUP) or (group == "mixed" and i < n + FULL_WAV_PER_GROUP) or \
               (group == "ssl" and i < n + n_mixed + FULL_WAV_PER_GROUP)
        samples.append({"index": i, "group": group, "label": [float(v) for v in label.tolist()], "ssl_label": int(ssl_label),
                        "is_mixed": bool(is_mixed), "original_single_label": int(orig), "ssl_task": str(ssl_task),
                        **_wav_summary(wav, full)})
    ds_test = ConcatenationMultilabelDataset(train_flag=False, transform=None, args=a, print_flag=False)
    test_samples = []
    for i in [0, 1, 2, 1000]:
        wav, label, *_ = ds_test[i]
        test_samples.append({"index": i, "filename": ds_test.all_cycles[i]["filename"], "label": [float(v) for v in label.tolist()],
                             "original_label": int(ds_test.all_cycles[i]["original_label"]), "wav": wav.reshape(-1).clone()})
    meta["test_n_cycles"] = len(ds_test.all_cycles)
    meta["test_cycle_filenames_head"] = [c["filename"] for c in ds_test.all_cycles[:20]]
    torch.save({"meta": meta, "raw_cycles": raw_cycles, "samples": samples, "test_samples": test_samples}, out / "dataset.pt")
    print("dataset.pt:", n, len(ds), meta["class_counts"], meta["short_counts"], "patients", meta["n_patients"])

    # fbank (AST input) — original generate_fbank → (T, F, 1) numpy
    w0 = samples[0]["wav"]
    w1 = next(s for s in samples if s["group"] == "mixed" and "wav" in s)["wav"]
    fb = torch.stack([torch.from_numpy(generate_fbank(w, 16000, 128)).squeeze(-1) for w in (w0, w1)])  # (2, 998, 128)
    torch.save({"wav": torch.stack([w0, w1]), "fbank": fb}, out / "fbank_input.pt")
    print("fbank_input.pt:", tuple(fb.shape))

    # SpecAugment (batch-level, p >= randn quirk)
    aug = SpecAugment(a)
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    x = fb.unsqueeze(1).clone()                   # (B, 1, T, F) — input convention of the original transforms
    y = aug(x.clone())
    torch.save({"input": x, "output": y, "seed": 3, "policy": "icbhi_ast_sup", "mask": "mean"}, out / "specaug.pt")
    print("specaug.pt: changed frac", float((y != x).float().mean()))

    # 3→4 conversion + Score
    combos = [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    conv = convert_original(a, np.array(combos))
    hits, counts = [10.0, 5.0, 3.0, 1.0], [20.0, 10.0, 6.0, 4.0]
    sp, se, sc, _ = get_score(hits, counts)
    torch.save({"combos": combos, "single": conv.tolist(), "hits": hits, "counts": counts, "sp": sp, "se": se, "score": sc}, out / "conversion.pt")
    print("conversion.pt:", dict(zip(map(tuple, combos), conv.tolist())), sp, se, sc)


def _loss_block(logits3, ssl_logits, labels3, ssl_labels):
    bce = nn.BCEWithLogitsLoss()(logits3, labels3)
    mask = ssl_labels >= 0
    ce = nn.CrossEntropyLoss()(ssl_logits[mask], ssl_labels[mask])
    return {"bce": bce.detach(), "ssl_ce": ce.detach(),
            "loss_w0.1": ((1 - 0.1) * bce + 0.1 * ce).detach(), "loss_w0.5": ((1 - 0.5) * bce + 0.5 * ce).detach(),
            "loss_paper_a0.1": (bce + 0.1 * ce).detach()}


def part_beats(args, out: pathlib.Path):
    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "models"))
    torch.cuda.is_available = lambda: False
    from BEATs.beats import BEATsTransferLearningModel  # noqa: E402

    g = torch.load(out / "fbank_input.pt", weights_only=False)
    wav = g["wav"]                                                  # (2, 160000)
    m = BEATsTransferLearningModel(num_target_classes=3, model_path=str(pathlib.Path(args.beats_ckpt).resolve()), spec_transform=None).eval()
    classifier, ssl_classifier = nn.Linear(768, 3), nn.Linear(768, 2)
    projector = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 768))
    for mod in (classifier, ssl_classifier, projector):
        for p in mod.parameters():
            det_fill(p)
    labels3 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ssl_labels = torch.tensor([-1, 1])
    with torch.no_grad():
        tokens = m(wav, training=False)
        feats = tokens.mean(dim=1)
        logits3 = classifier(feats)
        ssl_logits = ssl_classifier(feats)
        ssl_logits_sep = ssl_classifier(projector(feats))
    torch.save({"wav": wav, "tokens": tokens, "features": feats, "logits3": logits3, "ssl_logits": ssl_logits,
                "ssl_logits_separate": ssl_logits_sep, "labels3": labels3, "ssl_labels": ssl_labels,
                **_loss_block(logits3, ssl_logits, labels3, ssl_labels)}, out / "beats.pt")
    print("beats.pt:", tuple(tokens.shape), logits3, ssl_logits)


def part_ast(args, out: pathlib.Path):
    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "models"))
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    class _CPUDataParallel(nn.Module):
        def __init__(self, module, *a, **k):
            super().__init__()
            self.module = module

        def forward(self, *a, **k):
            return self.module(*a, **k)

    torch.nn.DataParallel = _CPUDataParallel
    # the original looks for ./pretrained_models/audioset_10_10_0.4593.pth relative to cwd (dropbox wget if missing) → run from a symlinked folder
    scratch = pathlib.Path(args.scratch) / "ast_cwd"
    (scratch / "pretrained_models").mkdir(parents=True, exist_ok=True)
    link = scratch / "pretrained_models" / "audioset_10_10_0.4593.pth"
    if not link.exists():
        link.symlink_to(pathlib.Path(args.ast_ckpt).resolve())
    os.chdir(scratch)
    import timm  # noqa: E402
    from AST.ast import ASTModel  # noqa: E402

    g = torch.load(out / "fbank_input.pt", weights_only=False)
    x = g["fbank"].unsqueeze(1)                                       # (2, 1, 998, 128) = batch after ToTensor (the same-size bilinear Resize(998,128) is treated as identity and skipped — not verified numerically)
    torch.manual_seed(0)
    model = ASTModel(label_dim=3, fstride=10, tstride=10, input_fdim=998, input_tdim=128, imagenet_pretrain=True,
                     audioset_pretrain=True, model_size="base384", verbose=False)
    classifier, ssl_classifier = nn.Linear(768, 3), nn.Linear(768, 2)
    for mod in (classifier, ssl_classifier):
        for p in mod.parameters():
            det_fill(p)
    model.eval()
    with torch.no_grad():
        feats = model(x, training=False).float()
        logits3 = classifier(feats)
        ssl_logits = ssl_classifier(feats)
    labels3 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ssl_labels = torch.tensor([-1, 1])
    torch.save({"input": x, "features": feats, "logits3": logits3, "ssl_logits": ssl_logits, "labels3": labels3, "ssl_labels": ssl_labels,
                "pos_embed_shape": list(model.v.pos_embed.shape), "f_dim_t_dim": list(model.get_shape(10, 10, 998, 128)),
                "v_keys": {k: list(v.shape) for k, v in model.v.state_dict().items()},
                **_loss_block(logits3, ssl_logits, labels3, ssl_labels)}, out / "ast.pt")
    json.dump({"repo": str(repo), "torch": torch.__version__, "timm": timm.__version__, "ckpt": args.ast_ckpt},
              open(out / "provenance_ast.json", "w"), indent=2)
    print("ast.pt:", tuple(feats.shape), logits3, model.v.pos_embed.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="outputs/reference/pcmcl")
    ap.add_argument("--out", default="tests/golden/pcmcl")
    ap.add_argument("--part", choices=["data", "beats", "ast"], required=True)
    ap.add_argument("--scratch", default="outputs/pcmcl/golden_scratch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--beats-ckpt", default="pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt")
    ap.add_argument("--ast-ckpt", default="pretrained_models/upstream/audioset_0.4593.pth")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.scratch = str((ROOT / args.scratch).resolve())
    args.repo = str((ROOT / args.repo).resolve())
    args.beats_ckpt = str((ROOT / args.beats_ckpt).resolve())
    args.ast_ckpt = str((ROOT / args.ast_ckpt).resolve())
    {"data": part_data, "beats": part_beats, "ast": part_ast}[args.part](args, out)
    prov = out / "provenance.json"
    d = json.load(open(prov)) if prov.exists() else {}
    d[args.part] = {"repo": args.repo, "torch": torch.__version__, "python": sys.version.split()[0], "seed": args.seed}
    json.dump(d, open(prov, "w"), indent=2)


if __name__ == "__main__":
    main()
