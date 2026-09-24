"""
Official-code parity golden and configuration tests for AMS-D (CcnNnn/AMS-D).

Golden source: scripts/reproduction/dump_amsd_goldens.py — the official models/ams_d.py, custom_loss.py, utils/metric.py run in the timm 0.4.5 reference venv (CPU)
(random parameters made deterministic with det_fill, real 16/16 ckpt loaded). augment.pt holds values from actually running the official loader/mydataset.py ICBHIDataset in .venv (synthetic wav, both loop/short padding paths).
Skipped if the golden/ckpt is missing.
"""
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rsc_toolkit.methods.amsd import AMSDLoss
from rsc_toolkit.metrics.icbhi.binary_score import PaperBinaryScore
from rsc_toolkit.metrics.icbhi.score import ICBHIScore


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "amsd" / "model.pt"
AUG = REPO / "tests" / "golden" / "amsd" / "augment.pt"
CKPT = REPO / "pretrained_models/upstream/audioset_16_16_0.4422.pth"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")
needs_aug = pytest.mark.skipif(not AUG.exists(), reason="augmentation golden missing")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="AudioSet 16/16 ckpt missing")
BUFFERS = {"all_predictions", "ids_masked", "current_epoch"}


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def _same_mask(ours, ref):
    """Same mask sets (equal after sorting) + order matches for at least 95% — only the argsort order of near-tied scores (1e-6 apart) fluctuates across timm versions."""
    return torch.equal(ours.sort(dim=1).values, ref.sort(dim=1).values) and (ours == ref).float().mean() > 0.95


def _build(**kw):
    from rsc_toolkit.models.ast_amsd import AMSDAST

    m = AMSDAST(n_cls=4, pretrained_path=str(CKPT), num_train_samples=8, **kw)
    for mod in (m.mlp_head, m.gpredlayer):
        for p in mod.parameters():
            _det_fill(p)
    for p in m.token_weight.parameters():  # same as the golden: non-degenerate sin fill (linspace makes all token scores tie)
        with torch.no_grad():
            p.copy_((torch.sin(torch.arange(p.numel(), dtype=torch.float32) * 0.7) * 0.05).reshape(p.shape))
    _det_fill(m.mask_embed)
    return m.eval()


# ------------------------------------------------------------------ loss / metrics (no golden needed)
def test_loss_double_softmax_teacher_epoch0():
    crit = AMSDLoss()
    s = torch.zeros(2, 4)
    t = torch.eye(4)[[0, 1]]
    total, ce, w, kl = crit(s, t, None, t, 0)
    teacher = F.softmax(t, 1)  # double softmax: [0.475,0.175,0.175,0.175]
    assert torch.allclose(teacher[0], torch.tensor([0.4754, 0.1749, 0.1749, 0.1749]), atol=1e-3)
    ref_kl = F.kl_div(F.log_softmax(s, 1), teacher, reduction="batchmean")
    assert torch.allclose(kl, ref_kl) and torch.allclose(ce, F.cross_entropy(s, torch.tensor([0, 1]))) and float(w) == 0
    assert torch.allclose(total, ce + kl)
    # assumption flag: use teacher directly as probabilities
    total2, _, _, kl2 = AMSDLoss(teacher_is_prob=True)(s, t, None, t, 0)
    assert torch.allclose(kl2, F.kl_div(F.log_softmax(s, 1), t, reduction="batchmean"))


def test_binary_metric_vs_standard():
    logits = F.one_hot(torch.tensor([0, 1, 2, 2, 0, 1, 0]), 4).float()
    target = torch.tensor([0, 0, 1, 2, 3, 1, 0])
    pb, st = PaperBinaryScore(), ICBHIScore(num_classes=4)
    pb.update(logits, target); st.update(logits, target)
    p, s = pb.compute(), st.compute()
    # binary: 3 of 4 abnormal have ŷ≠0 -> 75; 2 of 3 normal have ŷ=0 -> 66.67
    assert abs(float(p["paper_S_e"]) - 75.0) < 1e-4 and abs(float(p["paper_S_p"]) - 200 / 3) < 1e-4
    # standard 4-class: exact abnormal classes (1→2 ✗, 2→2 ✓, 3→0 ✗, 1→1 ✓) 2/4 = 50
    assert abs(float(s["S_e"]) - 50.0) < 1e-4 and abs(float(s["S_p"]) - float(p["paper_S_p"])) < 1e-4
    assert float(p["paper_score"]) > float(s["icbhi_score"])


@needs_golden
def test_binary_metric_matches_reference_cases():
    g = torch.load(GOLDEN, weights_only=False)
    for case in g["metric_cases"]:
        m = PaperBinaryScore()
        m.update(F.one_hot(torch.tensor(case["pred"]), 4).float(), torch.tensor(case["target"]))
        r = m.compute()
        assert abs(float(r["paper_S_e"]) / 100 - case["tpr"]) < 1e-6 and abs(float(r["paper_S_p"]) / 100 - case["tnr"]) < 1e-6
        assert abs(float(r["paper_acc"]) / 100 - case["acc"]) < 1e-6


# ------------------------------------------------------------------ data augmentation
@needs_aug
@pytest.mark.parametrize("case", ["loop", "short"])
def test_augment_pipeline_matches_official_dataset(case):
    """Toolkit pipeline (AudioPreprocessor -> MelBankAudioProcessor(raw, 798) -> AMSDFeatureDataset) = output of the official ICBHIDataset.__getitem__.
    loop = the paper's 8 s repeat padding (default assumption), short = code fallback (fbank 798 raw zero-padding; desired_length_sec=null pad_type=zero)."""
    import torchaudio

    from rsc_toolkit.datasets.amsd_data import AMSDFeatureDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor
    from rsc_toolkit.processors.mel import MelBankAudioProcessor

    g = torch.load(AUG, weights_only=False)
    c = g["cases"][case]
    wav = g["wav_short"]  # toolkit input is always a 2.7 s clip — for the loop case AudioPreprocessor repeat-pads to 8 s
    cfg = AudioPreprocessConfig(desired_length_sec=8.0, pad_type="repeat") if case == "loop" else AudioPreprocessConfig(None, pad_type="zero")
    audio = AudioPreprocessor(g["sr"], cfg)(wav)
    if case == "loop":
        assert torch.equal(audio, g["wav_loop"]) and audio.shape[1] == 8 * g["sr"]
    else:
        assert torch.equal(audio, wav)
    proc = MelBankAudioProcessor(n_mels=128, sampling_rate=g["sr"], target_frames=g["target_length"], frame_shift=10,
                                 window_type="hanning", dither=0.0, normalize=False, feature_cache=None)
    raw_ft = proc.make_features(audio)                        # (F, T) raw, zero-padded/truncated
    # the official _wav2fbank first subtracts the global waveform mean — kaldi remove_dc_offset (per frame) absorbs it mathematically, but in float32 a numeric difference of up to 2.4e-4
    # (values ~10, relative 2e-5) remains (0.0 when the global mean is subtracted). The shared MelBankAudioProcessor is left untouched; recorded as a tolerance.
    assert torch.allclose(raw_ft.transpose(0, 1), c["raw"], atol=5e-4), (raw_ft.transpose(0, 1) - c["raw"]).abs().max()
    if case == "short":
        assert c["n_frames_before_pad"] < 798 and torch.count_nonzero(c["raw"][c["n_frames_before_pad"]:]) == 0  # fallback zero-padding branch
    else:
        assert c["n_frames_before_pad"] == 798                                                                   # 8 s -> padding branch not taken

    ds = AMSDFeatureDataset.__new__(AMSDFeatureDataset)
    ds.training, ds.freqm, ds.timem, ds.mean, ds.std = True, g["freqm"], g["timem"], g["mean"], g["std"]
    ds.freq_mask = torchaudio.transforms.FrequencyMasking(g["freqm"] // 2)
    ds.time_mask = torchaudio.transforms.TimeMasking(g["timem"] // 2)
    torch.manual_seed(g["mask_seed"])
    train_tf = ds.transform(raw_ft).transpose(0, 1)           # (T, F): masking -> normalization
    assert torch.allclose(train_tf, c["train_nonoise"], atol=1e-4), (train_tf - c["train_nonoise"]).abs().max()  # fbank difference above /(2·std)
    ds.training = False
    assert torch.allclose(ds.transform(raw_ft).transpose(0, 1), c["eval"], atol=1e-4)
    if case == "short":  # padded region value = (0-mean)/(2std) (normalization after raw zero-padding — differs from the toolkit default 'zero-padding after normalization')
        assert torch.allclose(c["eval"][-1], torch.full((128,), (0 - g["mean"]) / (2 * g["std"])))
    # noise+roll: the torch part uses the same seed as the model's _noise_roll (2345+epoch+index); amplitude/shift replay the official global numpy RNG values from the golden
    gen = torch.Generator().manual_seed(2345 + g["epoch"] + g["index"])
    noised = c["train_nonoise"] + torch.rand(798, 128, generator=gen) * g["amp"]
    assert torch.allclose(torch.roll(noised, g["shift"], 0), c["train"], atol=1e-6)


# ------------------------------------------------------------------ model goldens
@needs_golden
@needs_ckpt
def test_model_matches_reference():
    g = torch.load(GOLDEN, weights_only=False)
    m = _build()
    d = g["dims"]
    assert (m.f_dim, m.t_dim, m.num_patches, m.num_masked) == (d["f_dim"], d["t_dim"], d["num_patches"], d["num_masked"]) == (8, 49, 392, 153)
    ours = {k: list(v.shape) for k, v in m.state_dict().items() if k not in BUFFERS and not k.startswith("criterion")}
    assert ours == g["state_keys"]
    assert torch.allclose(m.v.pos_embed, g["pos_embed"], atol=1e-6)
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["module.v.pos_embed"]
    cut = raw[:, 2:].reshape(1, 512, 768).transpose(1, 2).reshape(1, 768, 8, 64)[:, :, :, 8:57].reshape(1, 768, 392).transpose(1, 2)
    assert torch.allclose(m.v.pos_embed[:, 2:], cut)
    assert g["unfold_vs_conv_maxdiff"] < 1e-4  # (f) Unfold order = conv patch order

    x = g["input"]  # (2, 798, 128) time-freq
    with torch.no_grad():
        logits = m(x.transpose(-2, -1))  # toolkit batch (B,128,798) path
        assert torch.allclose(logits, g["eval_logits"], rtol=1e-4, atol=1e-3), (logits - g["eval_logits"]).abs().max()
        w0, mse0, ids0, s0 = m.amsd_forward(x, 0, None)
        assert float(mse0) == 0 and _same_mask(ids0, g["ep0"]["ids_masked"])
        assert torch.allclose(s0, g["ep0"]["s_out"], rtol=1e-4, atol=1e-3) and torch.allclose(w0, g["ep0"]["weighted_x"], rtol=1e-4, atol=1e-4)
        w1, mse1, ids1, s1 = m.amsd_forward(x, 1, g["ep1"]["ids_in"])
        assert torch.allclose(s1, g["ep1"]["s_out"], rtol=1e-4, atol=1e-3), (s1 - g["ep1"]["s_out"]).abs().max()
        assert torch.allclose(w1, g["ep1"]["weighted_x"], rtol=1e-4, atol=1e-4)
        assert torch.allclose(mse1, g["ep1"]["mse"], rtol=1e-4), (float(mse1), float(g["ep1"]["mse"]))
        assert _same_mask(ids1, g["ep1"]["ids_masked"])
        tokens = m.v.patch_embed(x.unsqueeze(1).transpose(2, 3))
        xm, _ = m.attentive_mask(tokens, g["ep1"]["ids_in"])
        assert torch.allclose(xm, g["ep1"]["x_masked"], atol=1e-5)
        # (g) other mask modes
        torch.manual_seed(g["masks"]["random_seed"])
        _, ids_rand = m.random_mask_patch(tokens.clone(), m.num_masked)
        assert torch.equal(ids_rand, g["masks"]["random"])
        assert torch.equal(m.time_interval_masking(tokens, m.t_dim, m.f_dim, m.num_masked)[1], g["masks"]["time_inter"])
        assert torch.equal(m.freq_interval_masking(tokens, m.t_dim, m.f_dim, m.num_masked)[1], g["masks"]["freq_inter"])
    # (d) Custom_Loss
    onehot = torch.eye(4)[g["labels"]]
    for key, teacher in (("onehot", onehot), ("prob", g["loss"]["t_prob"])):
        ours = m.criterion(g["ep1"]["s_out"], teacher, g["ep1"]["weighted_x"], onehot, g["ep1"]["mse"])
        for a, b in zip(ours, g["loss"][key]):
            assert torch.allclose(torch.as_tensor(a), b, rtol=1e-5), (key, float(a), float(b))
    ours0 = m.criterion(g["ep0"]["s_out"], onehot, g["ep0"]["weighted_x"], onehot, 0)
    for a, b in zip(ours0, g["loss"]["ep0"]):
        assert torch.allclose(torch.as_tensor(a), b, rtol=1e-5)


@needs_golden
@needs_ckpt
def test_training_forward_epoch_transition_and_buffers():
    """epoch 0: store one-hot teacher, no masking -> epoch 1: masking with stored ids, probability teacher (main.py:321-350)."""
    g = torch.load(GOLDEN, weights_only=False)
    m = _build().train()
    x = g["input"].transpose(-2, -1)  # (2,128,798)
    labels, idx = g["labels"], torch.tensor([3, 5])
    m.current_epoch.fill_(0)
    logits, loss = m.training_forward(x, labels, F.cross_entropy, meta_labels=idx)
    assert torch.allclose(logits, g["ep0"]["s_out"], rtol=1e-4, atol=1e-3)
    assert torch.allclose(m.all_predictions[idx], F.softmax(g["ep0"]["s_out"], 1), atol=1e-4)
    assert _same_mask(m.ids_masked[idx], g["ep0"]["ids_masked"]) and torch.count_nonzero(m.ids_masked[[0, 1, 2, 4, 6, 7]]) == 0
    assert torch.allclose(loss, g["loss"]["ep0"][0], rtol=1e-4)
    m.current_epoch.fill_(1)
    logits1, loss1 = m.training_forward(x, labels, F.cross_entropy, meta_labels=idx)
    assert torch.allclose(logits1, g["ep1"]["s_out"], rtol=1e-4, atol=1e-3)
    assert torch.allclose(loss1, g["loss"]["prob"][0], rtol=1e-4), (float(loss1), float(g["loss"]["prob"][0]))
    # validation path (no meta) = AST logits + CE, buffers unchanged
    m.eval()
    snap = m.all_predictions.clone()
    vl, vloss = m.training_forward(x, labels, F.cross_entropy)
    assert torch.allclose(vl, g["eval_logits"], rtol=1e-4, atol=1e-3) and torch.equal(snap, m.all_predictions)
    # buffers are included in state_dict (last.ckpt save/restore)
    assert BUFFERS <= set(m.state_dict())


@needs_ckpt
def test_method_variants_and_noise():
    torch.manual_seed(0)
    x = torch.randn(2, 128, 798)
    labels, idx = torch.tensor([0, 2]), torch.tensor([0, 1])
    for method in ("baseline", "sd_only", "sd_mask"):
        m = _build(method=method).train()
        m.current_epoch.fill_(1)
        logits, loss = m.training_forward(x, labels, F.cross_entropy, meta_labels=idx)
        assert logits.shape == (2, 4) and torch.isfinite(loss)
    m = _build(mask_mode="random").train()  # non-attentive: masking and MSE from epoch 0, no auxiliary
    with torch.no_grad():
        w, mse, ids, s = m.amsd_forward(x.transpose(-2, -1), 0, None)
    assert w is None and float(mse) > 0 and ids.shape == (2, 153)
    # noise/roll: deterministic for the same (epoch, index), different for a different index
    m.current_epoch.fill_(10)
    a = m._noise_roll(x.transpose(-2, -1), 10, idx)
    b = m._noise_roll(x.transpose(-2, -1), 10, idx)
    c = m._noise_roll(x.transpose(-2, -1), 10, idx + 1)
    assert torch.equal(a, b) and not torch.equal(a, c) and a.shape == (2, 798, 128)
