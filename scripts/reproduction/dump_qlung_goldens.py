"""
QLung (RSC-Toolkit/QLung 46408dd) official-code parity goldens — runs the official code as-is and saves the results to tests/golden/qlung/.

Each part runs in a different environment:
  aqs     — util/get_acoustic_features.py (needs librosa/scipy/matplotlib) → toolkit .venv (librosa 0.11):
              .venv/bin/python scripts/reproduction/dump_qlung_goldens.py aqs --repo outputs/reference/qlung
            Waveforms are ICBHI train cycles from the toolkit IndexDataset (16 kHz, 8 s repeat_fade — confirmed identical to the official cut_pad via the patchmix_pipeline goldens).
            Records rms / spectral entropy / three AQS variants (code as-is (rms,rms) · code formula + (entropy,rms) · paper formula).
  specaug — util/augmentation.py SpecAugment(icbhi_ast_sup) batch gate and masking (nlpaug stub) → .venv:
              .venv/bin/python scripts/reproduction/dump_qlung_goldens.py specaug --repo outputs/reference/qlung
  loss    — method/loss.py AngularClassifier / AngularMarginLoss(DFAM) on fixed tensors (det_fill W) → reference venv (timm 0.4.5, CPU):
              CUDA_VISIBLE_DEVICES= .venv-ref-timm045/bin/python scripts/reproduction/dump_qlung_goldens.py loss --repo outputs/reference/qlung
  model   — models/ast.py ASTModel (798/128 quirk, AudioSet ckpt) full forward + det_fill head logits and total loss → reference venv:
              CUDA_VISIBLE_DEVICES= .venv-ref-timm045/bin/python scripts/reproduction/dump_qlung_goldens.py model --repo outputs/reference/qlung
The hard-coded .cuda() calls in the official code (loss.py:39,55,60,64-75) are patched to identity (the torch in the reference venv cannot initialize CUDA on this server).
"""
import argparse
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

# CPU patches: .cuda() → identity, torch.load → CPU
torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
nn.Module.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

CLASS_NUMS = [2063, 1215, 501, 363]


def det_fill(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def _fixed_loss_inputs(dim: int = 768, n: int = 8):
    """Fixed features, labels and AQS. The last two rows are made opposite to W_d[y] (cos≈−1) to force the cos−mm branch of easy_margin=False."""
    g = torch.Generator().manual_seed(2026)
    feats = torch.randn(n, dim, generator=g)
    labels = torch.tensor([0, 1, 2, 3, 1, 0, 2, 3])
    aqs = torch.tensor([0.65, 0.40, 0.70, 0.55, 0.0, 1.0, 0.62, 0.45])
    return feats, labels, aqs


def _anti_aligned(feats, labels, weight):
    f = feats.clone()
    for i in (6, 7):
        f[i] = -weight[labels[i]].detach()
    return f


def dump_aqs(repo: pathlib.Path, out: pathlib.Path) -> None:
    # the matplotlib import at get_acoustic_features.py:4 is for plotting in __main__ — missing from the toolkit venv, so stubbed (unused on the computation path)
    for name in ("matplotlib", "matplotlib.pyplot"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, str(repo))
    from util.get_acoustic_features import (  # noqa: E402
        compute_quality_score_entropy_rms,
        compute_rms_energy,
        compute_spectral_entropy,
    )
    import librosa  # noqa: E402

    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    root = pathlib.Path.cwd()
    pre = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat_fade"))
    ds = IndexDataset(index_csv=root / ".datasets/icbhi/index/index_cycle.csv", data_root=root / ".datasets/icbhi",
                      sampling_rate=16000, sample_level="cycle", split="train",
                      split_file=root / "rsc_toolkit/datasets/icbhi/data/official_split.txt", audio_preprocessor=pre)
    picks = [0, 1, 100, 1500, 2777, 4000]
    rows = []
    for i in picks:
        item = ds[i]
        y = item["audio"].numpy().squeeze(0)
        rows.append((item["sample_id"], torch.from_numpy(y.copy())))
    g = torch.Generator().manual_seed(7)
    rows.append(("synthetic_randn", 0.1 * torch.randn(128000, generator=g)))
    rows.append(("synthetic_short_tone", torch.sin(torch.arange(24000, dtype=torch.float32) * 0.05)))

    golden = []
    for sid, wave in rows:
        y = wave.numpy().astype(np.float32)
        rms = compute_rms_energy(y)
        ent = compute_spectral_entropy(y)
        q_code_rms = compute_quality_score_entropy_rms(rms, rms, alpha=0.7, beta=0.3)          # icbhi_dataset.py:177 as-is
        q_code_entropy = compute_quality_score_entropy_rms(ent, rms, alpha=0.7, beta=0.3)      # code formula + actual entropy
        rms_score = float(np.clip((rms - 0.001) / (0.8 - 0.001), 0.0, 1.0))
        q_paper = float(np.clip(1.0 - 0.7 * ent + 0.3 * rms_score, 0.0, 1.0))                 # paper Eq. (5)
        golden.append({"sample_id": sid, "wave": wave, "rms": rms, "entropy": ent,
                       "aqs_code_rms": q_code_rms, "aqs_code_entropy": q_code_entropy, "aqs_paper": q_paper})
        print(f"{sid:45s} rms {rms:.6f} ent {ent:.6f} aqs(rms,rms) {q_code_rms:.6f} aqs(ent,rms) {q_code_entropy:.6f} paper {q_paper:.6f}")
    torch.save({"rows": golden, "librosa": librosa.__version__, "alpha": 0.7, "beta": 0.3}, out / "aqs_golden.pt")


def dump_specaug(repo: pathlib.Path, out: pathlib.Path) -> None:
    for name in ("nlpaug", "nlpaug.augmenter", "nlpaug.augmenter.audio"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, str(repo))
    from util.augmentation import SpecAugment  # noqa: E402

    args = types.SimpleNamespace(specaug_policy="icbhi_ast_sup", specaug_mask="mean")
    transforms = SpecAugment(args)
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    x = torch.randn(4, 1, 798, 128)  # [B, 1, time, freq] — original train batch (4 instead of bs 8)
    torch.save(x.clone(), out / "specaug_input.pt")
    outs = []
    for _ in range(6):  # several calls so that the gate (p ≥ randn) both passes and fails
        outs.append(transforms(x.clone()).clone())
    torch.save({"outs": outs}, out / "specaug_golden.pt")
    applied = [not torch.equal(o, x) for o in outs]
    print("specaug applied per call:", applied)


def dump_loss(repo: pathlib.Path, out: pathlib.Path) -> None:
    sys.path.insert(0, str(repo))
    from method.loss import AngularClassifier, AngularMarginLoss  # noqa: E402

    feats0, labels, aqs = _fixed_loss_inputs()
    clf = AngularClassifier(768, 4, s=37)
    dfam = AngularMarginLoss(768, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=CLASS_NUMS, easy_margin=False)
    det_fill(clf.weight)
    # W_d gets values different from the classifier
    with torch.no_grad():
        dfam.weight.copy_(torch.linspace(0.5, -0.5, dfam.weight.numel()).reshape(dfam.weight.shape).roll(97, dims=1))
    feats = _anti_aligned(feats0, labels, dfam.weight)
    feats_norm = F.normalize(feats)
    output = clf(feats_norm)
    class_loss = nn.CrossEntropyLoss()(output.to(torch.float64), labels)
    output2 = dfam(feats_norm, labels, aqs)
    aqm_loss = nn.CrossEntropyLoss()(output2, labels) * 0.4
    total = class_loss + aqm_loss
    dfam_easy = AngularMarginLoss(768, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=CLASS_NUMS, easy_margin=True)
    with torch.no_grad():
        dfam_easy.weight.copy_(dfam.weight)
    output2_easy = dfam_easy(feats_norm, labels, aqs)
    cosine = F.linear(feats_norm, F.normalize(dfam.weight))
    torch.save({
        "features": feats, "labels": labels, "aqs": aqs,
        "clf_weight": clf.weight.detach().clone(), "dfam_weight": dfam.weight.detach().clone(),
        "class_margin": dfam.class_margin.detach().clone(),
        "angular_logits": output.detach(), "dfam_logits": output2.detach(), "dfam_logits_easy": output2_easy.detach(),
        "dfam_cosine": cosine.detach(),
        "class_loss": class_loss.detach(), "aqm_loss": aqm_loss.detach(), "total_loss": total.detach(),
        "lambda": 0.4, "s_a": 37, "s_d": 15, "m": 0.5, "gamma": 0.5, "m_target": 0.2, "class_nums": CLASS_NUMS,
    }, out / "loss_golden.pt")
    print("class_margin", dfam.class_margin.tolist())
    print("dfam logits dtype", output2.dtype, "class_loss", float(class_loss), "aqm_loss", float(aqm_loss), "total", float(total))
    print("hard-branch rows (cos-th<=0):", ((cosine - torch.cos(torch.pi - dfam.gamma * dfam.m * aqs.double()
                                                                   - (1 - dfam.gamma) * dfam.class_margin[labels]).unsqueeze(1)) <= 0)
          .gather(1, labels.view(-1, 1)).squeeze(1).tolist())


def dump_model(repo: pathlib.Path, out: pathlib.Path) -> None:
    link = pathlib.Path("pretrained_models/audioset_10_10_0.4593.pth")
    target = pathlib.Path("pretrained_models/upstream/audioset_0.4593.pth")
    if not link.exists() and target.exists():
        os.symlink(target.resolve(), link)
    sys.path.insert(0, str(repo))
    from models.ast import ASTModel  # noqa: E402
    from method.loss import AngularClassifier, AngularMarginLoss  # noqa: E402
    import timm  # noqa: E402

    json.dump({"torch": torch.__version__, "timm": timm.__version__}, open(out / "reference_versions.json", "w"), indent=1)
    m = ASTModel(label_dim=4, fstride=10, tstride=10, input_fdim=798, input_tdim=128, imagenet_pretrain=True,
                 audioset_pretrain=True, model_size="base384", verbose=False)
    m.eval()
    x = torch.load("tests/golden/patchmix/features_input.pt")  # (2, 1, 798, 128) fbank
    with torch.no_grad():
        feats = m(x, training=False)
    clf = AngularClassifier(768, 4, s=37)
    dfam = AngularMarginLoss(768, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=CLASS_NUMS, easy_margin=False)
    det_fill(clf.weight)
    with torch.no_grad():
        dfam.weight.copy_(torch.linspace(0.5, -0.5, dfam.weight.numel()).reshape(dfam.weight.shape).roll(97, dims=1))
    labels = torch.tensor([0, 3])
    aqs = torch.tensor([0.6512, 0.4031])
    with torch.no_grad():
        feats_norm = F.normalize(feats)
        output = clf(feats_norm)
        class_loss = nn.CrossEntropyLoss()(output.to(torch.float64), labels)
        output2 = dfam(feats_norm, labels, aqs)
        aqm_loss = nn.CrossEntropyLoss()(output2, labels) * 0.4
    torch.save({"input": x, "features": feats.float(), "labels": labels, "aqs": aqs,
                "clf_weight": clf.weight.detach().clone(), "dfam_weight": dfam.weight.detach().clone(),
                "angular_logits": output, "dfam_logits": output2, "class_loss": class_loss, "aqm_loss": aqm_loss,
                "total_loss": class_loss + aqm_loss}, out / "model_golden.pt")
    json.dump({k: list(v.shape) for k, v in m.state_dict().items()}, open(out / "ast_state_dict_shapes.json", "w"),
              indent=0, sort_keys=True)
    print(f"feats {tuple(feats.shape)} mean {float(feats.mean()):+.6f}; total loss {float(class_loss + aqm_loss):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["aqs", "specaug", "loss", "model"])
    ap.add_argument("--repo", default="outputs/reference/qlung")
    ap.add_argument("--out", default="tests/golden/qlung")
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()
    {"aqs": dump_aqs, "specaug": dump_specaug, "loss": dump_loss, "model": dump_model}[args.part](repo, out)
    print("done:", out)


if __name__ == "__main__":
    main()
