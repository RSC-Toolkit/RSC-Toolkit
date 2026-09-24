"""
Lungmix (ICASSP 2025) — the augmentation function has official-code parity goldens (bit-identical to the official lungmix()); the rest are property checks of a re-implementation from the paper description (+ assumptions).

Golden: tests/golden/lungmix/lungmix_golden.pt (scripts/reproduction/dump_lungmix_goldens.py imports outputs/reference/lungmix/lungmix.py
and runs it with the same seed). Model tests that need the AudioSet AST ckpt are skipped without it.
"""
import pathlib

import numpy as np
import pytest
import torch

from rsc_toolkit.augmentations.lungmix import (
    ClassLayout, FIT_MODES, OVERLAP_MODES, fit_length, lungmix_batch, lungmix_official, lungmix_pair, or_label, or_labels,
    vanilla_mixup_batch,
)
from rsc_toolkit.methods.lungmix import LungmixLoss, mixup_ce
from rsc_toolkit.models.ast_patchmix import AUDIOSET_CKPT_CANDIDATES


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "lungmix" / "lungmix_golden.pt"
CKPT_AVAILABLE = any((REPO / c).exists() for c in AUDIOSET_CKPT_CANDIDATES)
SPRS_META = REPO / "rsc_toolkit" / "datasets" / "sprsound" / "data" / "metadata.csv"
SPRS_INDEX = REPO / ".datasets" / "sprsound" / "index" / "index_cycle.csv"
OFFICIAL_REF = REPO / "outputs" / "reference" / "lungmix" / "lungmix.py"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing — scripts/reproduction/dump_lungmix_goldens.py")
needs_ckpt = pytest.mark.skipif(not CKPT_AVAILABLE, reason="AudioSet AST ckpt missing")


@pytest.fixture(scope="module")
def golden():
    """The 6.3 MB golden is read only once per module."""
    return torch.load(GOLDEN, weights_only=False) if GOLDEN.exists() else {}


@pytest.fixture(autouse=True)
def _pin_threads(golden):
    """The golden was dumped with 1 thread — the last bit of the torch.mean(wav2) constant padding depends on the thread count (1.5e-10 apart between 64/16 and 1/4)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(int(golden.get("num_threads", 1)))
    yield
    torch.set_num_threads(prev)


# ------------------------------------------------------------------ official-code parity: official function golden
@needs_golden
def test_official_port_bit_exact(golden):
    g = golden
    assert len(g["cases"]) >= 4
    for c in g["cases"]:
        np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        wav, label, lam, l1, l2, w1, w2, mask, ml = lungmix_official(
            c["wav1"].clone(), c["wav2"].clone(), c["label1"], c["label2"], smooth=c["smooth"])
        assert torch.equal(wav, c["wav"]), c["name"]
        assert torch.equal(mask, c["mask"]) and torch.equal(w1, c["wav1_proc"]) and torch.equal(w2, c["wav2_proc"])
        assert label == c["label"] and lam == c["lam"] and ml == c["max_length"]
        assert wav.shape[1] >= max(c["wav1"].shape[1], c["wav2"].shape[1])  # length max_len+shift2 quirk


@needs_golden
@pytest.mark.skipif(not OFFICIAL_REF.exists(), reason="official clone (outputs/reference/lungmix) missing — only the golden dump comparison runs")
def test_official_file_live_matches_port(golden):
    """Imports the official lungmix.py live and checks bit-identity with the port and the golden under the same seed (only when the clone exists)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lungmix_official_ref", OFFICIAL_REF)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    for c in golden["cases"]:
        np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        o = official.lungmix(c["wav1"].clone(), c["wav2"].clone(), c["label1"], c["label2"], smooth=c["smooth"])
        np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        p = lungmix_official(c["wav1"].clone(), c["wav2"].clone(), c["label1"], c["label2"], smooth=c["smooth"])
        assert torch.equal(o[0], p[0]) and torch.equal(o[7], p[7]) and o[1] == p[1] and o[2] == p[2] and o[8] == p[8]
        assert torch.equal(o[0], c["wav"]), c["name"]


@needs_golden
def test_pair_code_mode_equals_official_and_paper_mode_applies_eq3(golden):
    g = golden
    for c in g["cases"]:
        np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        r = lungmix_pair(c["wav1"].clone(), c["wav2"].clone(), smooth=c["smooth"], apply_loudness_mask=False)
        assert torch.equal(r.wav, c["wav"]) and torch.equal(r.mask, c["mask"]) and r.lam == c["lam"]
        # same RNG sequence -> same mask, only the output uses the Eq. 3 blending
        np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        p = lungmix_pair(c["wav1"].clone(), c["wav2"].clone(), smooth=c["smooth"], apply_loudness_mask=True)
        assert torch.equal(p.mask, c["mask"])
        assert torch.equal(p.wav, p.mask * p.wav1 + (1 - p.mask) * p.wav2)
        vals = set(torch.unique(p.mask).tolist())
        # {0, λ, 1} — float32 rounding can give two values for λ (λ, (1+λ)−1) (official code quirk, kept as is)
        assert all(v == 0.0 or v == 1.0 or abs(v - p.lam) < 1e-6 for v in vals), vals
        if not c["smooth"]:
            assert 0.0 in vals and 1.0 in vals
        # the code version follows only the random mask even where loudness is on (differs from the paper version)
        rm = r.random_mask
        assert torch.equal(r.wav[rm], r.wav1[rm]) and torch.equal(r.wav[~rm], r.wav2[~rm])
        assert torch.equal(p.blend_mask, p.mask)                       # overlap='lam' = the mask returned by the official code, unchanged
        # overlap='one' (OR reading of Eq. 8): random positions = 1, loudness only = λ, rest 0; the returned mask is still the official mask
        np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        q = lungmix_pair(c["wav1"].clone(), c["wav2"].clone(), smooth=c["smooth"], apply_loudness_mask=True, overlap="one")
        assert torch.equal(q.mask, c["mask"]) and q.lam == p.lam
        exp = torch.where(q.random_mask, torch.ones_like(q.loudness_mask), q.loudness_mask * q.lam)
        assert torch.equal(q.blend_mask, exp.float())
        assert torch.equal(q.wav, q.blend_mask * q.wav1 + (1 - q.blend_mask) * q.wav2)
        assert bool((q.blend_mask[q.random_mask] == 1.0).all())
        both = q.random_mask & (q.loudness_mask > 0)
        if bool(both.any()):
            assert bool((p.blend_mask[both] < 1.0).all()) and bool((q.blend_mask[both] == 1.0).all())   # overlap: lam=λ vs one=1
    assert set(OVERLAP_MODES) == {"lam", "one"}
    with pytest.raises(ValueError):
        lungmix_pair(torch.zeros(1, 100), torch.zeros(1, 100), apply_loudness_mask=True, overlap="bogus")


# ------------------------------------------------------------------ label OR
def test_or_label_table_4x4():
    N, C, W, B = 0, 1, 2, 3
    table = {
        (N, N): N, (N, C): C, (N, W): W, (N, B): B,
        (C, N): C, (C, C): C, (C, W): B, (C, B): B,
        (W, N): W, (W, C): B, (W, W): W, (W, B): B,
        (B, N): B, (B, C): B, (B, W): B, (B, B): B,
    }
    for (a, b), exp in table.items():
        assert or_label(a, b) == exp
    names = ["normal", "crackle", "wheeze", "both"]
    from rsc_toolkit.augmentations.lungmix import _get_aug_label

    for a in range(4):
        for b in range(4):
            assert names[or_label(a, b)] == _get_aug_label(names[a], names[b])   # same as the official string function
    lay = ClassLayout(normal=3, crackle=0, wheeze=1, both=2)
    assert or_label(0, 1, lay) == 2 and or_label(3, 0, lay) == 0
    assert or_labels(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])).tolist() == [1, 3, 3]


# ------------------------------------------------------------------ batch = per sample
def test_batch_equals_per_sample_and_cut():
    wavs = torch.randn(4, 3000); labels = torch.tensor([0, 1, 2, 3]); idx = torch.tensor([2, 3, 0, 1])
    np.random.seed(7); torch.manual_seed(7)
    mb = lungmix_batch(wavs, labels, index=idx, out_len=3000)
    np.random.seed(7); torch.manual_seed(7)
    for i in range(4):
        r = lungmix_pair(wavs[i:i + 1], wavs[int(idx[i]):int(idx[i]) + 1])
        assert torch.equal(fit_length(r.wav, 3000).squeeze(0), mb.wav[i])
        assert abs(float(mb.lam[i]) - r.lam) < 1e-6   # float32 tensor vs float64
    assert mb.wav.shape == (4, 3000) and mb.y_or.tolist() == [2, 3, 2, 3] and mb.y_b.tolist() == [2, 3, 0, 1]
    assert fit_length(torch.arange(10.0).reshape(1, 10), 4).tolist() == [[0, 1, 2, 3]]
    assert fit_length(torch.arange(3.0).reshape(1, 3), 7).tolist() == [[0, 1, 2, 0, 1, 2, 0]]
    assert fit_length(torch.arange(10.0).reshape(1, 10), 4, "cut_last").tolist() == [[6, 7, 8, 9]]
    assert fit_length(torch.arange(3.0).reshape(1, 3), 7, "cut_last").tolist() == [[0, 1, 2, 0, 1, 2, 0]]
    np.random.seed(3)
    rc = fit_length(torch.arange(10.0).reshape(1, 10), 4, "random_crop")
    assert rc.shape == (1, 4) and rc[0, 1] == rc[0, 0] + 1
    assert fit_length(torch.arange(10.0).reshape(1, 10), 4, "none").shape == (1, 10)
    assert set(FIT_MODES) == {"cut_first", "cut_last", "random_crop", "none"}
    with pytest.raises(ValueError):
        fit_length(torch.zeros(1, 5), 3, "bogus")
    np.random.seed(1); torch.manual_seed(1)
    vm = vanilla_mixup_batch(wavs, labels, index=idx, alpha=0.5)
    lam = float(vm.lam[0])
    assert torch.allclose(vm.wav, lam * wavs + (1 - lam) * wavs[idx], atol=1e-6)


def test_batch_raw_stage_uses_valid_len_only():
    """mix_stage='raw' path: mixing cut to valid_len only = cutting first and calling lungmix_pair directly (the padded part does not enter mixing)."""
    L = 4000
    raw = torch.randn(3, L); vl = torch.tensor([1500, 4000, 2600]); labels = torch.tensor([1, 2, 0]); idx = torch.tensor([1, 2, 0])
    padded = raw.clone()
    for i in range(3):   # mimic 9 s repeat padding: repeats after valid
        n = int(vl[i]); padded[i, n:] = raw[i, :n].repeat(-(-L // n))[: L - n]
    np.random.seed(5); torch.manual_seed(5)
    mb = lungmix_batch(padded, labels, index=idx, out_len=L, valid_len=vl)
    np.random.seed(5); torch.manual_seed(5)
    for i in range(3):
        j = int(idx[i])
        r = lungmix_pair(padded[i:i + 1, : int(vl[i])], padded[j:j + 1, : int(vl[j])])
        assert torch.equal(fit_length(r.wav, L).squeeze(0), mb.wav[i])
        assert r.max_length <= max(int(vl[i]), int(vl[j])) * 2      # based on the original segment length (shift2 < len(wav1))
    assert mb.wav.shape == (3, L)
    # mixing without valid_len (padded stage) gives a different result — the padded region enters mixing
    np.random.seed(5); torch.manual_seed(5)
    mb2 = lungmix_batch(padded, labels, index=idx, out_len=L)
    assert not torch.equal(mb2.wav, mb.wav)
    with pytest.raises(ValueError):
        lungmix_batch(padded, labels, index=idx, out_len=L, valid_len=torch.tensor([1, 2]))


def test_lungmix_index_dataset_emits_n_valid():
    """LungmixIndexDataset puts the pre-padding segment length (n_valid) into the item, and the datamodule attaches it as meta for train only."""
    from rsc_toolkit.datasets.lungmix_data import LungmixIndexDataset, _BandpassThenPreprocess, valid_len_meta
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=9.0, pad_type="repeat"))
    w = _BandpassThenPreprocess(None, pre)
    assert w(torch.randn(1, 20000)).shape == (1, 144000) and w.last_raw_len == 20000
    assert w(torch.randn(1, 200000)).shape == (1, 144000) and w.last_raw_len == 200000
    assert valid_len_meta({"n_valid": 7}) == 7
    icbhi = REPO / ".datasets" / "icbhi"
    if not (icbhi / "index/index_cycle.csv").exists():
        pytest.skip("ICBHI index missing")
    ds = LungmixIndexDataset(index_csv=icbhi / "index/index_cycle.csv", data_root=icbhi, sampling_rate=16000,
                             sample_level="cycle", split="train", resample_after_slice=True, fade=False,
                             split_file=REPO / "rsc_toolkit/datasets/icbhi/data/official_split.txt", audio_preprocessor=pre,
                             bandpass={"enabled": True}, label_map="none")
    it = ds[0]
    assert it["audio"].shape[-1] == 144000 and 0 < it["n_valid"] <= 144000
    assert it["n_valid"] == min(144000, ds._pre.last_raw_len)


def test_sprsound_label_map_none_rejected():
    from rsc_toolkit.datasets.lungmix_data import LungmixIndexDataset

    if not SPRS_INDEX.exists():
        pytest.skip("SPRSound index missing")
    kw = dict(index_csv=SPRS_INDEX, data_root=REPO / ".datasets" / "sprsound", sampling_rate=16000, sample_level="cycle",
              split="train", time_unit="ms_int")
    with pytest.raises(ValueError, match="sprsound_7to4"):
        LungmixIndexDataset(label_map="none", **kw)
    ds = LungmixIndexDataset(label_map="sprsound_7to4", **kw)
    assert int(ds.df["label"].max()) == 3 and "label_original" in ds.df


# ------------------------------------------------------------------ band-pass
def test_bandpass_frequency_response_and_shape():
    from rsc_toolkit.processors.bandpass import BandpassFilter

    bp = BandpassFilter(16000, 50.0, 1500.0, order=4, zero_phase=True)
    resp = bp.frequency_response([300.0, 800.0, 1000.0, 10.0, 3000.0, 6000.0])
    assert (resp[:3] > 0.9).all(), resp        # passband (|H|^2, zero-phase)
    assert (resp[3:] < 0.05).all(), resp       # attenuation below 50 Hz and above 1500 Hz
    x = torch.randn(1, 16000)
    y = bp(x)
    assert y.shape == x.shape and y.dtype == torch.float32
    # tone check: 500 Hz passes, 4 kHz attenuated
    t = torch.arange(16000) / 16000.0
    tone_in = torch.sin(2 * np.pi * 500 * t).reshape(1, -1); tone_out = torch.sin(2 * np.pi * 4000 * t).reshape(1, -1)
    assert bp(tone_in)[:, 2000:-2000].pow(2).mean() > 0.4 and bp(tone_out)[:, 2000:-2000].pow(2).mean() < 0.01


def test_lungmix_index_dataset_bandpass_before_pad(tmp_path):
    """Preprocessing wrapper of LungmixIndexDataset: band-pass -> 9 s repeat padding (before mixing, before padding)."""
    from rsc_toolkit.datasets.lungmix_data import _BandpassThenPreprocess
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor
    from rsc_toolkit.processors.bandpass import BandpassFilter

    bp = BandpassFilter(16000); pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=9.0, pad_type="repeat"))
    x = torch.randn(1, 20000)
    y = _BandpassThenPreprocess(bp, pre, before_pad=True)(x)
    assert y.shape == (1, 144000)
    assert torch.equal(y, pre(bp(x)))


# ------------------------------------------------------------------ SPRSound 7→4
def test_sprsound_7to4_mapping_counts():
    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4, SPRSOUND_EVENT_ID_TO_NAME, sprsound_4class_counts

    # Rhonchi(5) and Stridor(6) are excluded from the 4-class view — not in the map
    assert SPRSOUND_7_TO_4 == {0: 0, 1: 1, 2: 2, 3: 1, 4: 3}
    assert 5 not in SPRSOUND_7_TO_4 and 6 not in SPRSOUND_7_TO_4
    counts = sprsound_4class_counts(SPRS_META)   # bundled metadata (original label distribution -> pins the 4-class numbers)
    assert counts["train"] == {0: 5159, 1: 961, 2: 452, 3: 30}   # wheeze 452 = Rhonchi 39 and Stridor 15 excluded
    assert counts["valid"] == {0: 1040, 1: 83, 2: 305, 3: 1}
    if SPRS_INDEX.exists():
        import pandas as pd

        idx = pd.read_csv(SPRS_INDEX); meta = pd.read_csv(SPRS_META)
        by_name = meta["Label"].value_counts().to_dict(); by_id = idx["label"].value_counts().to_dict()
        # the 7 label counts are all distinct, so the id<->name mapping is uniquely determined
        assert len(set(by_name.values())) == 7
        for i, name in SPRSOUND_EVENT_ID_TO_NAME.items():
            assert by_id[i] == by_name[name], (i, name)


def test_apply_label_map_drops_excluded_rows():
    import pandas as pd
    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4, apply_label_map, map_labels

    df = pd.DataFrame({"sample_id": list("abcdefg"), "label": [0, 1, 2, 3, 4, 5, 6]})
    out, dropped = apply_label_map(df, SPRSOUND_7_TO_4)
    assert dropped == 2 and out["sample_id"].tolist() == list("abcde")
    assert out["label"].tolist() == [0, 1, 2, 1, 3] and out["label_original"].tolist() == [0, 1, 2, 3, 4]
    assert list(out.index) == [0, 1, 2, 3, 4]                       # reset_index — __getitem__(iloc) assumes a contiguous index
    assert len(df) == 7 and "label_original" not in df.columns      # original df unchanged
    assert map_labels([5, 6], SPRSOUND_7_TO_4) == [None, None]


# ------------------------------------------------------------------ loss variants
def test_loss_variants():
    torch.manual_seed(0)
    logits = torch.randn(6, 4); y_a = torch.tensor([0, 1, 2, 3, 1, 2]); y_b = torch.tensor([1, 2, 0, 0, 1, 3])
    y_or = or_labels(y_a, y_b); lam = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7, 0.0])
    ce = torch.nn.CrossEntropyLoss(); ce_none = torch.nn.CrossEntropyLoss(reduction="none")
    assert torch.allclose(LungmixLoss("non_linear")(logits, y_or, y_a, y_b, lam, ce), ce(logits, y_or))
    assert torch.allclose(LungmixLoss("none")(logits, y_or, y_a, y_b, lam, ce), ce(logits, y_a))
    assert torch.allclose(LungmixLoss("label_preservation")(logits, y_or, y_a, y_b, lam, ce), ce(logits, y_a))
    exp_mix = (lam * ce_none(logits, y_a) + (1 - lam) * ce_none(logits, y_b)).mean()
    assert torch.allclose(LungmixLoss("linear")(logits, y_or, y_a, y_b, lam, ce), exp_mix)
    assert torch.allclose(LungmixLoss("vanilla_mixup")(logits, y_or, y_a, y_b, lam, ce), exp_mix)
    assert torch.allclose(mixup_ce(logits, y_a, y_b, 1.0, ce), ce(logits, y_a))
    l_ce = ce(logits, y_or)
    comb = LungmixLoss("combined")(logits, y_or, y_a, y_b, lam, ce)
    assert torch.allclose(comb, l_ce + (l_ce.detach() / exp_mix.detach()) * exp_mix)   # λ2 = L_CE/L_mix (assumption)
    assert torch.allclose(comb, 2 * l_ce)                                               # numerically both terms have the same magnitude
    fixed = LungmixLoss("combined", lambda2_mode="fixed", lambda2=0.5)(logits, y_or, y_a, y_b, lam, ce)
    assert torch.allclose(fixed, l_ce + 0.5 * exp_mix)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0]); cew = torch.nn.CrossEntropyLoss(weight=w)
    assert torch.allclose(LungmixLoss("non_linear")(logits, y_or, y_a, y_b, lam, cew), cew(logits, y_or))


# ------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def model():
    from rsc_toolkit.models.factory import ModelFactory

    return ModelFactory.get_model_class("ast_lungmix")(n_cls=4)


@needs_ckpt
def test_model_layout_pooling_bn_params(model):
    import torch.nn as nn

    assert model.v.pos_embed.shape == (1, 2 + 12 * 101, 768)      # original AST 128×1024 -> no interpolation
    assert (model.f_dim, model.t_dim) == (12, 101)
    assert isinstance(model.bn, nn.BatchNorm1d) and model.bn.num_features == 768
    assert isinstance(model.head, nn.Linear) and model.head.out_features == 4
    assert isinstance(model.mlp_head, nn.Identity) and isinstance(model.classifier, nn.Identity)
    n_v = sum(p.numel() for p in model.v.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    assert n_all == n_v + 2 * 768 + 768 * 4 + 4
    assert n_all == 87_729_876
    model.eval()
    with torch.no_grad():
        fb = torch.randn(2, 1024, 128)
        tokens = model.forward_tokens(fb)
        assert tokens.shape == (2, 1214, 768)
        assert torch.allclose(model.forward_features(fb), tokens.mean(1))
        model.pool = "mean_patches"; assert torch.allclose(model.forward_features(fb), tokens[:, 2:].mean(1))
        model.pool = "cls_dist"; assert torch.allclose(model.forward_features(fb), (tokens[:, 0] + tokens[:, 1]) / 2)
        model.pool = "mean_all"
        assert torch.allclose(model(fb), model.head(model.bn(tokens.mean(1))))
    assert isinstance(model.ln, nn.Identity) and model.head_mode == "bn_linear"


@needs_ckpt
def test_model_head_bn_ln_linear_and_stride_flags():
    import torch.nn as nn

    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_lungmix")(n_cls=4, head="bn_ln_linear")
    assert isinstance(m.ln, nn.LayerNorm)
    assert sum(p.numel() for p in m.parameters()) == 87_729_876 + 2 * 768     # + LayerNorm(768) γ/β
    m.eval()
    with torch.no_grad():
        fb = torch.randn(1, 1024, 128)
        assert torch.allclose(m(fb), m.head(m.ln(m.bn(m.forward_features(fb)))))
    assert (m.fstride, m.tstride) == (10, 10)
    with pytest.raises(ValueError):
        ModelFactory.get_model_class("ast_lungmix")(n_cls=4, head="bogus")
    with pytest.raises(ValueError):
        ModelFactory.get_model_class("ast_lungmix")(n_cls=4, mix_stage="bogus")
    with pytest.raises(ValueError):
        ModelFactory.get_model_class("ast_lungmix")(n_cls=4, overlap="bogus")


@needs_ckpt
def test_fbank_9s_shape_and_padding(model):
    wav = torch.randn(2, 144000)   # 9 s @ 16 kHz
    fb = model.compute_fbank(wav)
    assert fb.shape == (2, 1024, 128)
    from rsc_toolkit.processors.mel import generate_fbank

    raw = generate_fbank(wav[:1], 16000, n_mels=128, normalize=False)
    assert raw.shape[0] == 898 and raw.shape[0] < 1024               # 9 s -> 898 frames + zero-padding
    pad_val = (0.0 - model.fbank_mean) / (model.fbank_std * 2)
    assert torch.allclose(fb[0, 898:], torch.full((1024 - 898, 128), pad_val))   # normalization after padding (assumption)
    assert torch.allclose(fb[0, :898], (raw - model.fbank_mean) / (model.fbank_std * 2), atol=1e-5)
    model.fbank_pad_before_norm = False
    fb2 = model.compute_fbank(wav)
    assert torch.equal(fb2[0, 898:], torch.zeros(1024 - 898, 128)) and torch.allclose(fb2[0, :898], fb[0, :898])
    model.fbank_pad_before_norm = True
    assert model.compute_fbank(torch.randn(1, 200000)).shape == (1, 1024, 128)   # long input is truncated


@needs_ckpt
def test_training_forward_variants_and_eval_path(model):
    ce = torch.nn.CrossEntropyLoss()
    wav = torch.randn(2, 144000) * 0.1; y = torch.tensor([1, 2])
    model.eval()
    with torch.no_grad():
        logits, loss = model.training_forward(wav, y, ce)
        assert logits.shape == (2, 4) and torch.allclose(loss, ce(logits, y))
    model.train()
    assert model.mix_stage == "raw"          # default, main run (mixing original segments)
    model.mix_stage = "padded"
    for v in ("none", "non_linear", "combined", "lungmix_patchmix"):
        model.variant = v; model.lungmix_loss.variant = v
        torch.manual_seed(0); np.random.seed(0)
        logits, loss = model.training_forward(wav, y, ce)
        assert logits.shape == (2, 4) and torch.isfinite(loss)
    # lungmix_patchmix label-preserving mode = CE(y_or)
    model.patchmix_label_mode = "preserve"
    torch.manual_seed(0); np.random.seed(0)
    idx = torch.randperm(2)                      # the pair lungmix_batch draws first (mix_prob=1 -> no preceding torch.rand)
    torch.manual_seed(0); np.random.seed(0)
    logits, loss = model.training_forward(wav, y, ce)
    assert torch.allclose(loss, ce(logits, or_labels(y, y[idx])))
    model.patchmix_label_mode = "mixup"
    model.variant = model.lungmix_loss.variant = "non_linear"
    # mix_stage raw: error without meta_labels; with them, only original segments are mixed
    model.mix_stage = "raw"
    with pytest.raises(ValueError, match="emit_valid_len"):
        model.training_forward(wav, y, ce)
    torch.manual_seed(0); np.random.seed(0)
    logits, loss = model.training_forward(wav, y, ce, meta_labels=torch.tensor([30000, 50000]))
    assert logits.shape == (2, 4) and torch.isfinite(loss)
    # training forward works for both apply_loudness_mask paths and both overlap modes (raw stage)
    model.apply_loudness_mask = True
    for ov in OVERLAP_MODES:
        model.overlap = ov
        torch.manual_seed(0); np.random.seed(0)
        logits, loss = model.training_forward(wav, y, ce, meta_labels=torch.tensor([30000, 50000]))
        assert torch.isfinite(loss)
    model.overlap = "lam"
    model.apply_loudness_mask = False
    model.mix_stage = "padded"
    torch.manual_seed(0); np.random.seed(0)
    logits, loss = model.training_forward(wav, y, ce)
    assert torch.isfinite(loss)
    model.mix_stage = "raw"


@needs_ckpt
def test_specaugment_in_model_optional():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_lungmix")(n_cls=4, specaugment={"enabled": True, "policy": "icbhi_ast_sup"})
    fb = torch.randn(2, 1024, 128)
    m.train(); torch.manual_seed(0); np.random.seed(0)
    out = m.apply_specaugment(fb.clone())
    assert out.shape == fb.shape and not torch.equal(out, fb)
    m.eval()
    assert torch.equal(m.apply_specaugment(fb.clone()), fb)
    m0 = ModelFactory.get_model_class("ast_lungmix")(n_cls=4)
    m0.train()
    assert m0.specaugment is None and torch.equal(m0.apply_specaugment(fb.clone()), fb)
