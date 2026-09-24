"""
GaP-Aug (ICASSP 2024) — tests for a re-implementation from the paper description.

Only the backbone (PANNs Cnn14) can be compared with goldens (scripts/reproduction/dump_gapaug_goldens.py: official audioset_tagging_cnn + torchlibrosa, 16k ckpt, 512/160).
GaP-Aug, mixup, triplet and the balanced sampler are property checks against the paper description (+ assumptions).
"""
import pathlib

import pytest
import torch

from rsc_toolkit.augmentations.gap_aug import GammaPatchAug, SpecAugPANNs, normalize_nonneg, sample_patches
from rsc_toolkit.datasets.gapaug_data import BalancedBatchSampler
from rsc_toolkit.methods.gapaug import mixup_pairs, pair_nll, random_triplets, soft_target_nll, split_odd_even, triplet_loss


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "gapaug" / "backbone.pt"
CKPT = REPO / "pretrained_models" / "upstream" / "Cnn14_16k_mAP=0.438.pth"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="PANNs Cnn14 16k ckpt missing")


def _model(**kw):
    from rsc_toolkit.models.cnn14_gapaug import Cnn14GaPAug

    kw.setdefault("pretrained", "16k" if CKPT.exists() else None)
    return Cnn14GaPAug(**kw)


# ------------------------------------------------------------------ backbone golden
@needs_golden
@needs_ckpt
def test_backbone_matches_official_panns():
    g = torch.load(GOLDEN, weights_only=False)
    m = _model(method="gapaug", feature_preset="panns16k", norm_mode="none").eval()
    with torch.no_grad():
        lm = m.compute_logmel(g["wav"])
        assert lm.shape == g["logmel"].shape == (3, 1, 1001, 64)
        d = (lm - g["logmel"]).abs()
        # float32 noise from the STFT implementation difference (conv1d vs torch.stft): max ~4e-3 dB, mean ~7e-5 dB (float64 recomputation also 2e-3)
        assert d.max() < 1e-2 and d.mean() < 1e-3, (d.max().item(), d.mean().item())
        out = m.forward_backbone(lm)
        assert torch.allclose(out["feat"], g["embedding"], atol=1e-4), (out["feat"] - g["embedding"]).abs().max().item()
        assert torch.equal(out["emb"], out["feat"])                                # eval: no dropout -> emb == feat
        assert out["pooled"].shape == (3, 2048)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]      # fc_audioset is not in the golden (avoids redistribution)
        fc = torch.nn.Linear(2048, 527)
        fc.weight.data.copy_(ck["fc_audioset.weight"]); fc.bias.data.copy_(ck["fc_audioset.bias"])
        assert torch.allclose(fc(out["emb"]), g["logits"], atol=1e-4)


@needs_golden
@needs_ckpt
def test_checkpoint_key_coverage_and_param_count():
    g = torch.load(GOLDEN, weights_only=False)
    m = _model(method="gapaug")
    info = m.pretrained_info
    official = {k for k in g["state_keys"] if not k.startswith(("spectrogram_extractor.", "logmel_extractor.", "fc_audioset."))}
    assert set(m.backbone.state_dict()) == official                   # everything except the excluded keys matches (strict load)
    assert info["n_loaded"] == len(official)
    assert set(info["excluded"]) == {k for k in g["state_keys"] if k.startswith(("spectrogram_extractor.", "logmel_extractor.", "fc_audioset."))}
    for k, v in m.backbone.state_dict().items():
        assert list(v.shape) == g["state_shapes"][k], k
    assert sum(v.numel() for v in m.backbone.state_dict().values()) == g["n_params_wo_audioset_and_frontend"]
    n_params = sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for p in m.fc.parameters()) == 2048 * 4 + 4
    # PANNs Cnn14 parameters (excluding buffers) 80,753,615 − fc_audioset 1,079,823 = 79,673,792 ; + fc(4) 8,196 = 79,681,988
    assert n_params == 79_681_988


def test_strict_load_rejects_missing_key(tmp_path):
    if not CKPT.exists():
        pytest.skip("ckpt missing")
    state = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    del state["fc1.weight"]
    torch.save({"model": state}, tmp_path / "bad.pth")
    from rsc_toolkit.models.cnn14_gapaug import Cnn14GaPAug

    with pytest.raises(RuntimeError):
        Cnn14GaPAug(pretrained=None, pretrained_path=str(tmp_path / "bad.pth"))


def test_paper_feature_preset_shapes():
    m = _model(method="naive", pretrained=None).eval()
    assert (m.n_fft, m.hop) == (400, 100)
    lm = m.compute_logmel(torch.randn(2, 160000))
    assert lm.shape == (2, 1, 1601, 64)
    x = m.make_input(torch.randn(2, 160000))
    assert x.min() >= 0 and x.max() <= 1.0 + 1e-6


# ------------------------------------------------------------------ GaP-Aug
def test_gap_aug_patches_gamma_and_untouched_regions():
    torch.manual_seed(0)
    x = torch.rand(3, 1, 200, 64)
    aug = GammaPatchAug(n_patches=32, max_freq=16, max_time=16, gamma_range=(1.7, 2.0), gain=1.0)
    out, info = aug(x, return_info=True)
    assert out.shape == x.shape and (out >= 0).all()
    for b, inf in enumerate(info):
        p, g = inf["patches"], inf["gamma"]
        assert p.shape == (32, 4) and g.shape == (32,)
        assert ((g >= 1.7) & (g <= 2.0)).all()
        h_t, h_f = p[:, 1] - p[:, 0], p[:, 3] - p[:, 2]
        assert ((h_t >= 1) & (h_t <= 16)).all() and ((h_f >= 1) & (h_f <= 16)).all()
        assert (p[:, 0] >= 0).all() and (p[:, 1] <= 200).all() and (p[:, 2] >= 0).all() and (p[:, 3] <= 64).all()
        mask = torch.zeros(200, 64, dtype=torch.bool)
        for k in range(32):
            t0, t1, f0, f1 = (int(v) for v in p[k])
            mask[t0:t1, f0:f1] = True
        assert torch.equal(out[b, 0][~mask], x[b, 0][~mask])          # unchanged outside patches
        assert (out[b, 0][mask] <= x[b, 0][mask] + 1e-6).all()          # on [0,1], γ>1 decreases values (suppresses weak signals)
        assert not torch.equal(out[b, 0][mask], x[b, 0][mask])


def test_gap_aug_single_patch_is_power():
    x = torch.rand(1, 1, 32, 32)
    aug = GammaPatchAug(n_patches=1, max_freq=32, max_time=32, min_freq=32, min_time=32, gamma_range=(1.8, 1.8))
    out = aug(x)
    assert torch.allclose(out, x.pow(1.8))


def test_patchmask_zeroes_patches_and_specaug_shape():
    torch.manual_seed(1)
    x = torch.rand(2, 1, 100, 64) + 0.1
    out, info = GammaPatchAug(n_patches=8, mode="mask")(x, return_info=True)
    p = info[0]["patches"]
    t0, t1, f0, f1 = (int(v) for v in p[0])
    assert (out[0, 0, t0:t1, f0:f1] == 0).all()
    assert (out > 0).float().mean() < 1.0
    sa = SpecAugPANNs()
    y = sa(x)
    assert y.shape == x.shape and (y == 0).any()


def test_normalize_modes():
    x = torch.randn(4, 1, 50, 64) * 30 - 40
    s = normalize_nonneg(x, "sample_minmax")
    assert torch.allclose(s.amin(dim=(1, 2, 3)), torch.zeros(4), atol=1e-6) and torch.allclose(s.amax(dim=(1, 2, 3)), torch.ones(4), atol=1e-6)
    b = normalize_nonneg(x, "batch_minmax")
    assert b.min() >= 0 and b.max() <= 1 + 1e-6
    d = normalize_nonneg(x, "db_offset", db_offset=100.0)
    assert d.min() >= 0
    assert torch.equal(normalize_nonneg(x, "none"), x)


# ------------------------------------------------------------------ mixup / triplet
def test_mixup_pairs_and_soft_labels():
    torch.manual_seed(0)
    idx_a, idx_b = split_odd_even(16)
    assert idx_a.tolist() == list(range(0, 16, 2)) and idx_b.tolist() == list(range(1, 16, 2))
    x = torch.rand(16, 1, 10, 8); y = torch.randint(0, 4, (16,))
    xm, ym, lam = mixup_pairs(x[idx_a], x[idx_b], y[idx_a], y[idx_b], 4, alpha=1.0)
    assert xm.shape == (8, 1, 10, 8) and ym.shape == (8, 4)
    assert torch.allclose(ym.sum(1), torch.ones(8), atol=1e-6)
    assert ((lam >= 0) & (lam <= 1)).all()
    for i in range(8):
        exp = lam[i] * x[idx_a[i]] + (1 - lam[i]) * x[idx_b[i]]
        assert torch.allclose(xm[i], exp)
        assert torch.isclose(ym[i, y[idx_a[i]]] + (0 if y[idx_a[i]] != y[idx_b[i]] else 0), ym[i, y[idx_a[i]]])
    logits = torch.randn(8, 4)
    assert torch.isclose(soft_target_nll(logits, ym), pair_nll(logits, y[idx_a], y[idx_b], lam), atol=1e-5)
    # α<=0 → λ=1, x_mix = x_a
    xm0, ym0, lam0 = mixup_pairs(x[idx_a], x[idx_b], y[idx_a], y[idx_b], 4, alpha=0.0)
    assert torch.equal(xm0, x[idx_a]) and (lam0 == 1).all()


def test_triplet_loss_finite_and_valid_indices():
    torch.manual_seed(0)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    a, p, n = random_triplets(y)
    assert len(a) == 8 and (y[a] == y[p]).all() and (y[a] != y[n]).all() and (a != p).all()
    emb = torch.randn(8, 2048, requires_grad=True)
    loss = triplet_loss(emb, y, margin=1.0)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward(); assert emb.grad is not None
    # classes without positives are excluded; all the same class gives 0
    assert triplet_loss(torch.randn(3, 8), torch.tensor([0, 0, 0])).item() == 0.0
    a2, _, _ = random_triplets(torch.tensor([0, 1, 2, 2]))
    assert a2.tolist() == [2, 3]


# ------------------------------------------------------------------ training forward
def _tiny(**kw):
    return _model(pretrained=None, **kw)


def test_training_forward_methods_run():
    torch.manual_seed(0)
    wav = torch.randn(8, 16000)  # shortened to 1 s (same structure)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    ce = torch.nn.CrossEntropyLoss()
    for method in ("gapaug", "gapaug_nomixup", "naive", "specaug", "patchmask", "mixup_only"):
        m = _tiny(method=method).train()
        logits, loss = m.training_forward(wav, y, ce)
        assert logits.shape == (8, 4) and torch.isfinite(loss), method
        loss.backward()
    m = _tiny(method="gapaug").eval()
    logits, loss = m.training_forward(wav, y, ce)
    assert torch.isclose(loss, ce(logits, y))


def test_naive_equals_no_augmentation():
    """naive: no augmentation -> with the same seed augment() is the identity and the loss is CE + w_trip·triplet."""
    torch.manual_seed(0)
    m = _tiny(method="naive", w_trip=0.0).train()
    x = torch.rand(4, 1, 50, 64)
    assert torch.equal(m.augment(x), x)
    wav = torch.randn(4, 16000); y = torch.tensor([0, 1, 2, 3])
    torch.manual_seed(1); logits, loss = m.training_forward(wav, y, torch.nn.CrossEntropyLoss())
    torch.manual_seed(1); out = m.forward_backbone(m.make_input(wav))
    assert torch.allclose(logits, out["logits"]) and torch.isclose(loss, torch.nn.functional.cross_entropy(out["logits"], y))
    assert m.use_mixup is False and m.aug_kind == "none"
    g = _tiny(method="gapaug")
    assert g.use_mixup is True and g.aug_kind == "gamma"
    assert _tiny(method="specaug", mixup=False).use_mixup is False


def test_mixup_lambda_from_global_rng_is_reproducible():
    torch.manual_seed(3); _, _, l1 = mixup_pairs(torch.rand(4, 2), torch.rand(4, 2), torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long), 4)
    torch.manual_seed(3); _, _, l2 = mixup_pairs(torch.rand(4, 2), torch.rand(4, 2), torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long), 4)
    assert torch.equal(l1, l2)


# ------------------------------------------------------------------ balanced sampler
def test_balanced_sampler_batch_distribution():
    labels = [0] * 100 + [1] * 30 + [2] * 10 + [3] * 3
    s = BalancedBatchSampler(labels, per_class=4, steps_per_epoch=20, n_classes=4, seed=0)
    assert len(s) == 20 and s.batch_size == 16
    batches = list(s)
    assert len(batches) == 20
    lab = torch.tensor(labels)
    for b in batches:
        assert len(b) == 16
        counts = torch.bincount(lab[b], minlength=4).tolist()
        assert counts == [4, 4, 4, 4], counts
    # class 3 has only 3 samples -> sampling with replacement allows duplicates
    assert any(len(set(i for i in b if labels[i] == 3)) < 4 for b in batches)
    # a different sequence per epoch, reproducible with a fixed seed
    b2 = list(s)
    assert b2 != batches
    s2 = BalancedBatchSampler(labels, per_class=4, steps_per_epoch=20, n_classes=4, seed=0)
    assert list(s2) == batches


def test_sample_patches_bounds():
    p = sample_patches(100, T=10, F=64, max_t=16, max_f=16, min_t=1, min_f=1, gen=torch.Generator().manual_seed(0))
    assert (p[:, 1] <= 10).all() and (p[:, 1] - p[:, 0] >= 1).all() and (p[:, 3] - p[:, 2] <= 16).all()


# ------------------------------------------------------------------ option flags (head input, triplet layer, window, normalization)
def test_head_input_and_triplet_layer_flags():
    """head_input(emb|feat), triplet_layer(emb|feat|pooled), window(hann|hamming) flags actually change the path."""
    torch.manual_seed(0)
    wav = torch.randn(4, 16000)
    x = _tiny(method="naive").make_input(wav)
    m_emb = _tiny(method="naive", head_input="emb").eval()
    m_feat = _tiny(method="naive", head_input="feat").eval()
    m_feat.load_state_dict(m_emb.state_dict())
    with torch.no_grad():
        assert torch.allclose(m_emb.forward_backbone(x)["logits"], m_feat.forward_backbone(x)["logits"])  # eval: no dropout -> identical
    m_emb.train(); m_feat.train()
    torch.manual_seed(5); le = m_emb.forward_backbone(x)["logits"]
    torch.manual_seed(5); lf = m_feat.forward_backbone(x)["logits"]
    assert not torch.allclose(le, lf)                                                  # train: emb has the dropout mask applied before the head
    for layer, dim in (("emb", 2048), ("feat", 2048), ("pooled", 2048)):
        m = _tiny(method="naive", triplet_layer=layer).train()
        out = m.forward_backbone(x)
        assert m._trip(out).shape == (4, dim) and m._trip(out) is out[layer]
    with torch.no_grad():
        out = _tiny(method="naive").eval().forward_backbone(x)
        assert not torch.equal(out["pooled"], out["feat"])
    with pytest.raises(ValueError):
        _tiny(head_input="bogus")
    with pytest.raises(ValueError):
        _tiny(triplet_layer="bogus")
    with pytest.raises(ValueError):
        _tiny(window="blackman")
    h = _tiny(method="naive", window="hamming").eval()
    assert h.frontend.window == "hamming"
    lm_hann = _tiny(method="naive").eval().compute_logmel(wav)
    lm_hamm = h.compute_logmel(wav)
    assert lm_hann.shape == lm_hamm.shape and not torch.allclose(lm_hann, lm_hamm)
    assert h.describe()["window"] == "hamming" and h.describe()["head_input"] == "emb"


def test_gap_aug_norm_none_keeps_outside_patches():
    """With norm_mode=none (negative inputs) patches leave the outside unchanged too; inside, clamp(min=0)^γ."""
    torch.manual_seed(2)
    x = torch.randn(2, 1, 100, 64) * 20 - 30
    aug = GammaPatchAug(n_patches=8, gamma_range=(1.7, 2.0))
    out, info = aug(x, return_info=True)
    for b, inf in enumerate(info):
        mask = torch.zeros(100, 64, dtype=torch.bool)
        for k in range(8):
            t0, t1, f0, f1 = (int(v) for v in inf["patches"][k])
            mask[t0:t1, f0:f1] = True
        assert torch.equal(out[b, 0][~mask], x[b, 0][~mask])
        assert (out[b, 0][mask] >= 0).all()


def test_mixup_lambda_is_beta_and_pretrained_spec_has_sha256():
    torch.manual_seed(0)
    _, _, lam = mixup_pairs(torch.rand(2000, 1), torch.rand(2000, 1), torch.zeros(2000, dtype=torch.long), torch.ones(2000, dtype=torch.long), 4, alpha=1.0)
    assert abs(lam.mean().item() - 0.5) < 0.03 and abs(lam.var().item() - 1 / 12) < 0.01      # Beta(1,1) = U[0,1]
    _, _, lam2 = mixup_pairs(torch.rand(2000, 1), torch.rand(2000, 1), torch.zeros(2000, dtype=torch.long), torch.ones(2000, dtype=torch.long), 4, alpha=0.2)
    assert lam2.var().item() > lam.var().item()                                                 # α<1 -> mass pushed to the extremes
    from rsc_toolkit.models.cnn14_gapaug import PANNS_CNN14
    assert PANNS_CNN14["16k"]["sha256"] == "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
    assert PANNS_CNN14["16k"]["size"] == 358_668_570
    g = torch.load(GOLDEN, weights_only=False) if GOLDEN.exists() else {}
    assert "fc_audioset.weight" not in g                                                        # avoids redistribution


# ------------------------------------------------------------------ defaults, normalization restore, provenance, sampler resume
def test_defaults_follow_algorithm1_text():
    """Defaults = paper text: w_trip 1.0 (Algo. 1 line 14, unweighted sum), joint_forward False (line 13, two forwards). 0.01 is a flag."""
    m = _tiny(method="gapaug")
    assert m.w_trip == 1.0 and m.joint_forward is False
    assert _tiny(method="gapaug", w_trip=0.01).w_trip == 0.01
    torch.manual_seed(0)
    wav = torch.randn(8, 16000); y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    ce = torch.nn.CrossEntropyLoss()
    m2 = _tiny(method="gapaug", joint_forward=True).train()
    m2.load_state_dict(m.state_dict())
    m.train()
    _, l1 = m.training_forward(wav, y, ce)
    _, l2 = m2.training_forward(wav, y, ce)
    assert torch.isfinite(l1) and torch.isfinite(l2)


def test_norm_mode_sample_minmax_restore():
    """sample_minmax_restore: evaluation input stays on the original dB scale (=none); training augmentation applies γ on [0,1] then restores -> outside patches equal the original dB values."""
    from rsc_toolkit.augmentations.gap_aug import sample_minmax_restore, sample_minmax_split

    torch.manual_seed(0)
    x = torch.randn(3, 1, 80, 64) * 20 - 40
    x01, mn, mx = sample_minmax_split(x)
    assert torch.allclose(x01.amin(dim=(1, 2, 3)), torch.zeros(3), atol=1e-6) and x01.max() <= 1 + 1e-6
    assert torch.allclose(sample_minmax_restore(x01, mn, mx), x, atol=1e-4)
    assert torch.equal(normalize_nonneg(x, "sample_minmax_restore"), x)
    wav = torch.randn(2, 16000)
    m = _tiny(method="gapaug", norm_mode="sample_minmax_restore").train()
    lm = m.compute_logmel(wav)
    assert torch.equal(m.make_input(wav), lm)                          # model input (evaluation) = original dB
    torch.manual_seed(4); xa = m.make_augmented_input(wav)
    assert xa.shape == lm.shape and (xa <= lm + 1e-3).all()            # γ>1 -> values only move toward min
    changed = (xa - lm).abs() > 1e-2                                   # ignore float noise (~1e-5) from the restore round trip
    assert 0 < changed.float().mean() < 0.5                            # only inside patches changes (32 patches ≤16×16 / 1601×64)
    assert abs(xa.min().item() - lm.min().item()) < 1e-3
    with pytest.raises(ValueError):
        _tiny(norm_mode="bogus")


def test_golden_provenance_records_panns_commit():
    meta = pathlib.Path(GOLDEN).with_name("meta.json")
    prov = pathlib.Path(GOLDEN).with_name("provenance.json")
    if not meta.exists():
        pytest.skip("golden missing")
    import json
    md = json.loads(meta.read_text())
    assert md.get("fc_audioset_in_golden") is False and md.get("panns_commit") and md.get("torchlibrosa")
    if prov.exists():
        pv = json.loads(prov.read_text())
        assert pv["panns_commit"] == md["panns_commit"] and pv["panns_license"] == "MIT"


def test_balanced_sampler_set_epoch_matches_sequential_epoch():
    labels = [0] * 40 + [1] * 30 + [2] * 20 + [3] * 10
    seq = BalancedBatchSampler(labels, per_class=4, steps_per_epoch=5, n_classes=4, seed=0)
    ep = [list(seq) for _ in range(3)]  # run epochs 0,1,2 sequentially
    resumed = BalancedBatchSampler(labels, per_class=4, steps_per_epoch=5, n_classes=4, seed=0)
    resumed.set_epoch(2)  # resume: the callback sets current_epoch=2
    assert [b.tolist() if hasattr(b, "tolist") else list(b) for b in list(resumed)] == [b.tolist() if hasattr(b, "tolist") else list(b) for b in ep[2]]
    assert ep[0] != ep[2]  # not a replay of the epoch-0 sequence
