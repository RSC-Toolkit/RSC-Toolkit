"""
AMS-D (CcnNnn/AMS-D 5f8aab9) official-code parity goldens — runs the official models/{ams_d,ast_model,token_weight_module}.py, custom_loss.py and utils/metric.py
in the timm 0.4.5 reference venv (CPU only) and saves the results to tests/golden/amsd/.

  # (1) model/loss/metric goldens (reference venv: timm 0.4.5, torch CPU — CUDA cannot be initialized, so is_available is patched to False)
  CUDA_VISIBLE_DEVICES= .venv-ref-timm045/bin/python scripts/reproduction/dump_amsd_goldens.py \
      --repo outputs/reference/amsd --ckpt pretrained_models/upstream/audioset_16_16_0.4422.pth --out tests/golden/amsd
  # (2) data augmentation goldens (the reference venv has no torchaudio, so the toolkit .venv is used; official loader/mydataset.py:104-129)
  .venv/bin/python scripts/reproduction/dump_amsd_goldens.py --aug-only --out tests/golden/amsd

Goldens (model.pt):
- the official AMSD (label_dim 4, 16/16, 128×798, imagenet+audioset, mask_ratio 0.39, mask_dense 0, attentive) built with the real 16/16 ckpt.
  Randomly initialized parameters mlp_head, gpredlayer and mask_embed are made deterministic with det_fill(linspace), token_weight with sin_fill (non-degenerate).
  (a) state_dict keys/shapes, the truncated pos_embed (1,394,768) and its slice [8:57] of the original ckpt pos_embed
  (b) eval-path forward logits / epoch 0 train path (no masking): weighted_x, mse=0, ids_masked (for the next epoch), s_out
  (c) epoch 1 train path (given ids_masked = the ids from (b)): weighted_x, mse, new ids_masked, s_out
  (d) Custom_Loss values — teacher as one-hot (epoch 0) / probabilities (softmax logits), per component
  (e) utils/metric.py calculate_stats binary Se/Sp cases
  (f) max error between the Unfold patch order and the conv patch order (patch_embed output = unfold·W^T + b)
  (g) random/time_inter/freq_inter mask indices (random uses manual_seed(123))
Goldens (augment.pt, --aug-only): original fbank (raw, 700×128) → zero-padding to 798 → FrequencyMasking(24)·TimeMasking(96) (seed 7) → normalization → epoch 12·index 5 noise+roll.
"""
import argparse
import shutil
import json
import os
import pathlib
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def sin_fill(t):
    """Deterministic non-degenerate fill for TokenWeights — with linspace all token scores collapse to 63.6669±6e-6, so the argsort order depends on numerical noise."""
    with torch.no_grad():
        t.copy_((torch.sin(torch.arange(t.numel(), dtype=torch.float32) * 0.7) * 0.05).reshape(t.shape))


def dump_aug(out: pathlib.Path, repo: pathlib.Path):
    """Augmentation goldens from **actually running** the official loader/mydataset.py ICBHIDataset (.venv: depends only on torch/torchaudio/numpy).

    Puts a synthetic wav (16 kHz) at the temporary path <out>/_augwork/<case>/<label>/clip.wav and calls __getitem__(0) in train (mode, noise, epoch 12) and eval.
      case 'short': the 2.7 s clip as-is → code fallback path (raw zero-padding of the fbank to 798 frames)  (toolkit: desired_length_sec=null, pad_type=zero)
      case 'loop' : wav with the same clip repeat-padded to 8 s → paper §3.1 loop-pad path (fbank exactly 798 frames) (toolkit: 8.0, repeat)
    Randomness: masking is called right after torch.manual_seed(MASK_SEED); noise amplitude and roll right after np.random.seed(NP_SEED) → replayed in the same order (rand → randint) and recorded.
    """
    import torchaudio

    sys.path.insert(0, str(repo))
    from loader.mydataset import ICBHIDataset

    MASK_SEED, NP_SEED, EPOCH = 7, 11, 12
    target_length, freqm, timem = 798, 48, 192
    mean, std = -4.2677393, 4.5689974
    sr = 16000
    torch.manual_seed(0)
    wav_short = (torch.randn(1, int(2.7 * sr)) * 0.1).clamp(-1, 1)
    reps = -(-8 * sr // wav_short.shape[1])
    wav_loop = wav_short.repeat(1, reps)[:, : 8 * sr]  # paper loop-pad = toolkit AudioPreprocessor(pad_type=repeat)
    golden = {"sr": sr, "wav_short": wav_short, "wav_loop": wav_loop, "mask_seed": MASK_SEED, "np_seed": NP_SEED, "epoch": EPOCH,
              "index": 0, "freqm": freqm, "timem": timem, "mean": mean, "std": std, "target_length": target_length,
              "torchaudio": torchaudio.__version__, "cases": {}}
    np.random.seed(NP_SEED)
    amp, shift = float(np.random.rand()) / 10, int(np.random.randint(-10, 10))  # same call order as mydataset.py:128-129
    golden["amp"], golden["shift"] = amp, shift
    for case, wav in (("short", wav_short), ("loop", wav_loop)):
        root = out / "_augwork" / case
        (root / "1").mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(root / "1" / "clip.wav"), wav, sr)
        wav_disk, sr_disk = torchaudio.load(str(root / "1" / "clip.wav"))   # values after 16-bit PCM quantization = the waveform the official code actually reads
        assert sr_disk == sr
        golden["wav_short" if case == "short" else "wav_loop"] = wav_disk
        conf = dict(num_mel_bins=128, target_length=target_length, freqm=freqm, timem=timem, mixup=0, dataset="icbhi",
                    mode="train", mean=mean, std=std, noise=True, num_classes=4)
        ds = ICBHIDataset(str(root), conf, noise=True, epoch=EPOCH)
        assert len(ds) == 1 and ds.data[0][1] == 1
        raw = ds._wav2fbank(str(root / "1" / "clip.wav"))                      # (798, 128) raw (including zero-padding)
        torch.manual_seed(MASK_SEED); np.random.seed(NP_SEED)
        train_out, label, idx = ds[0]                                          # masking → normalization → noise+roll
        ds_nonoise = ICBHIDataset(str(root), {**conf, "noise": False}, noise=False, epoch=EPOCH)
        torch.manual_seed(MASK_SEED)
        train_nonoise, _, _ = ds_nonoise[0]                                    # masking → normalization (before noise)
        ds_eval = ICBHIDataset(str(root), {**conf, "mode": "eval"}, noise=False, epoch=EPOCH)
        eval_out, _, _ = ds_eval[0]                                            # normalization only
        assert tuple(raw.shape) == (798, 128) and int(label) == 1 and idx == 0
        golden["cases"][case] = {"raw": raw, "train_nonoise": train_nonoise, "train": train_out, "eval": eval_out,
                                 "n_frames_before_pad": int(torchaudio.compliance.kaldi.fbank(
                                     wav - wav.mean(), htk_compat=True, sample_frequency=sr, use_energy=False, window_type="hanning",
                                     num_mel_bins=128, dither=0.0, frame_shift=10).shape[0])}
        print(f"augment[{case}]: raw frames before pad {golden['cases'][case]['n_frames_before_pad']}, "
              f"masked ratio {(train_nonoise == (0 - mean) / (2 * std)).float().mean():.3f}")
    assert golden["cases"]["loop"]["n_frames_before_pad"] == 798 and golden["cases"]["short"]["n_frames_before_pad"] < 798
    assert torch.equal(golden["wav_short"].repeat(1, reps)[:, : 8 * sr], golden["wav_loop"])  # quantization is deterministic per sample → commutes with repeat padding
    shutil.rmtree(out / "_augwork", ignore_errors=True)
    torch.save(golden, out / "augment.pt")
    print("saved augment.pt (values from running the official ICBHIDataset)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="outputs/reference/amsd")
    ap.add_argument("--ckpt", default="pretrained_models/upstream/audioset_16_16_0.4422.pth")
    ap.add_argument("--out", default="tests/golden/amsd")
    ap.add_argument("--aug-only", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.aug_only:
        dump_aug(out, pathlib.Path(args.repo).resolve())
        return

    repo = pathlib.Path(args.repo).resolve()
    ckpt = pathlib.Path(args.ckpt).resolve()
    x = torch.load("tests/golden/patchmix/features_input.pt").squeeze(1)  # (2, 798, 128) = (time, freq)
    assert tuple(x.shape) == (2, 798, 128)

    # --- the torch (cu130) in the reference venv cannot initialize CUDA on this server → force-CPU patches (same approach as the addrsc goldens)
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    nn.Module.cuda = lambda self, *a, **k: self  # Custom_Loss calls nn.LogSoftmax(dim=1).cuda(), criterion.cuda(gpu)

    class _CPUDataParallel(nn.Module):
        def __init__(self, module, *a, **k):
            super().__init__()
            self.module = module

        def forward(self, *a, **k):
            return self.module(*a, **k)

    torch.nn.DataParallel = _CPUDataParallel
    sys.modules["wget"] = types.ModuleType("wget")  # wget is missing in the reference venv (not called when the ckpt exists)
    # sklearn is missing in the reference venv — utils/metric.py only uses accuracy_score (4-class acc) → replaced with numpy
    skl = types.ModuleType("sklearn"); skm = types.ModuleType("sklearn.metrics")
    skm.accuracy_score = lambda y, p: float(np.mean(np.asarray(y) == np.asarray(p)))
    import importlib.machinery as _mach
    skl.__spec__ = _mach.ModuleSpec("sklearn", None); skm.__spec__ = _mach.ModuleSpec("sklearn.metrics", None)  # for dynamo find_spec
    skl.metrics = skm; sys.modules["sklearn"] = skl; sys.modules["sklearn.metrics"] = skm

    # the official code reads 'my_kd/pretrained_models/audioset_16_16_0.4422.pth' relative to cwd → symlinked work folder
    work = out / "_work"
    (work / "my_kd" / "pretrained_models").mkdir(parents=True, exist_ok=True)
    link = work / "my_kd" / "pretrained_models" / "audioset_16_16_0.4422.pth"
    if not link.exists():
        link.symlink_to(ckpt)
    sys.path.insert(0, str(repo))
    os.chdir(work)
    import timm
    from models.ams_d import AMSD
    from custom_loss import Custom_Loss
    from utils.metric import calculate_stats

    torch.manual_seed(0)
    net = AMSD(label_dim=4, fshape=16, tshape=16, fstride=16, tstride=16, input_fdim=128, input_tdim=798,
               imagenet_pretrain=True, audioset_pretrain=True, model_size="base384", verbose=True,
               mask_ratio=0.39, mask_dense=0, mask_mode="attentive")
    assert (net.f_dim, net.t_dim, net.num_patches, net.num_masked) == (8, 49, 392, 153)
    for mod in (net.mlp_head, net.gpredlayer):
        for p in mod.parameters():
            det_fill(p)
    for p in net.token_weight.parameters():
        sin_fill(p)
    det_fill(net.mask_embed)
    net.eval()  # dropout 0 / pos_drop 0, so the numbers match train mode; eval for determinism

    sd_raw = torch.load(str(ckpt), map_location="cpu")
    pos_raw = sd_raw["module.v.pos_embed"]  # (1, 514, 768)
    labels = torch.tensor([0, 3])
    onehot = torch.eye(4)[labels]

    with torch.no_grad():
        # (b) eval path + epoch 0 train path
        eval_logits = net(x, None, None, mode="eval")[0]
        w0, mse0, ids0, s0 = net(x, 0, None, mode="train")
        assert mse0 == 0
        # (c) epoch 1 train path (ids_masked = ids0)
        w1, mse1, ids1, s1 = net(x, 1, ids0, mode="train")
        # (f) unfold order vs conv order
        inp = x.unsqueeze(1).transpose(2, 3)  # (B,1,128,798)
        tokens = net.v.patch_embed(inp)  # (B,392,768)
        unf = net.unfold(inp).transpose(1, 2)  # (B,392,256)
        w = net.v.patch_embed.proj.weight.reshape(768, 256)
        recon = unf @ w.t() + net.v.patch_embed.proj.bias
        unfold_vs_conv = (recon - tokens).abs().max().item()
        # (g) other mask modes
        pe = tokens.clone()
        torch.manual_seed(123)
        _, ids_rand = net.random_mask_patch(pe.clone(), net.num_masked)
        _, ids_time = net.time_interval_masking(pe.clone(), net.t_dim, net.f_dim, net.num_masked)
        _, ids_freq = net.freq_interval_masking(pe.clone(), net.t_dim, net.f_dim, net.num_masked)
        # masked tokens (attentive_mask) — hard masking result
        xm, _ = net.attentive_mask(pe.clone(), ids0)

    # (d) Custom_Loss — teacher one-hot / probabilities
    crit = Custom_Loss(distillation_type="soft", tau=1.0, alpha=1.0, beta=3e-2, gamma=3e-1)
    t_prob = F.softmax(s0, dim=1)
    with torch.no_grad():
        l_oh = crit(s1, onehot, w1, onehot, mse1)
        l_pr = crit(s1, t_prob, w1, onehot, mse1)
        l_e0 = crit(s0, onehot, w0, onehot, 0)  # epoch 0: mse 0 (int)
    # (e) binary metric
    metric_cases = []
    for tgt, pred in [([0, 0, 1, 2, 3, 1, 0], [0, 1, 2, 2, 0, 1, 0]), ([1, 2, 3], [3, 1, 2]), ([0, 0], [1, 0]), ([0, 1, 2, 3], [0, 1, 2, 3])]:
        st = calculate_stats(np.eye(4)[pred], np.eye(4)[tgt])
        metric_cases.append({"target": tgt, "pred": pred, "tpr": st["tpr"], "tnr": st["tnr"], "acc": st["acc"]})

    torch.save(
        {
            "input": x, "labels": labels,
            "state_keys": {k: list(v.shape) for k, v in net.state_dict().items()},
            "pos_embed": net.v.pos_embed.detach().clone(), "pos_embed_raw_shape": list(pos_raw.shape),
            "pos_cut_slice": [8, 57],
            "dims": {"f_dim": net.f_dim, "t_dim": net.t_dim, "num_patches": net.num_patches, "num_masked": net.num_masked},
            "eval_logits": eval_logits,
            "ep0": {"weighted_x": w0, "ids_masked": ids0, "s_out": s0},
            "ep1": {"ids_in": ids0, "weighted_x": w1, "mse": torch.as_tensor(mse1), "ids_masked": ids1, "s_out": s1, "x_masked": xm},
            "loss": {"onehot": [torch.as_tensor(v) for v in l_oh], "prob": [torch.as_tensor(v) for v in l_pr],
                     "ep0": [torch.as_tensor(v) for v in l_e0], "t_prob": t_prob},
            "metric_cases": metric_cases,
            "unfold_vs_conv_maxdiff": unfold_vs_conv,
            "masks": {"random_seed": 123, "random": ids_rand, "time_inter": ids_time, "freq_inter": ids_freq},
        },
        out / "model.pt",
    )
    json.dump({"repo": str(repo), "ckpt": str(ckpt), "torch": torch.__version__, "timm": timm.__version__,
               "note": "forced CPU (cuda.is_available=False, nn.Module.cuda no-op, DataParallel replacement, wget stub); random parameters made deterministic with det_fill for mlp_head/gpredlayer/mask_embed and sin_fill for token_weight (with linspace all token scores tie). augment.pt holds values from actually running the official loader/mydataset.py ICBHIDataset in .venv (synthetic wav)"},
              open(out / "provenance.json", "w"), indent=2)
    print("eval_logits", eval_logits, "\ns1", s1, "\nmse1", float(mse1), "loss", [float(v) for v in l_pr],
          "unfold_vs_conv", unfold_vs_conv, "ids0[0,:5]", ids0[0, :5].tolist())


if __name__ == "__main__":
    main()
