"""
Official-code parity golden and configuration tests for PC-MCL (arXiv 2601.17080, wa976/PC-MCL).

Golden source: scripts/reproduction/dump_pcmcl_goldens.py — results of running the official util/dataset_pcmcl.py, icbhi_util.py, augmentation.py, method/conversion.py,
models/BEATs (real iter3+ AS2M ckpt) and models/AST (timm 0.4.5 reference venv, AudioSet 10/10 ckpt, head det_fill) with minimal patches.
Tests are skipped when the golden/ckpt/ICBHI index is missing.
"""
import pathlib
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsc_toolkit.methods.pcmcl import (
    PMHead,
    combine_losses,
    label4_to_multilabel,
    multilabel_binary_to_label4,
    multilabel_logits_to_onehot4,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "pcmcl"
ICBHI = REPO / ".datasets" / "icbhi" / "index" / "index_cycle.csv"
SPLIT = REPO / "rsc_toolkit/datasets/icbhi/data/official_split.txt"
BEATS_CKPT = REPO / "pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt"
AST_CKPT = REPO / "pretrained_models/upstream/audioset_0.4593.pth"

needs = lambda name: pytest.mark.skipif(not (GOLDEN / name).exists(), reason=f"golden {name} missing")  # noqa: E731
needs_icbhi = pytest.mark.skipif(not ICBHI.exists(), reason="ICBHI index missing")
needs_beats = pytest.mark.skipif(not BEATS_CKPT.exists(), reason="BEATs ckpt missing")
needs_ast = pytest.mark.skipif(not AST_CKPT.exists(), reason="AudioSet AST ckpt missing")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


@pytest.fixture(scope="module")
def train_ds():
    if not ICBHI.exists():
        pytest.skip("ICBHI index missing")
    from rsc_toolkit.datasets.pcmcl_concat import PCMCLConcatDataset

    return PCMCLConcatDataset(index_csv=ICBHI, data_root=REPO / ".datasets/icbhi", split_file=SPLIT, split="train",
                              desired_length_sec=10.0, mixing_prob=1.0, enable_ssl=True, ssl_prob=0.3, rng_seed=1,
                              cache_dir=REPO / "outputs/pcmcl/cache")


# ------------------------------------------------------------------ label conversion/loss (no golden needed)
def test_label_conversion_rules():
    y4 = torch.tensor([0, 1, 2, 3])
    assert torch.equal(label4_to_multilabel(y4), torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]))
    combos = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    # conversion.py additive: [1,0,0]->0, c&w->3, c->1, w->2, all 0->0, normal+c->1, normal+w->2
    assert multilabel_binary_to_label4(combos).tolist() == [0, 2, 1, 3, 0, 2, 1, 3]
    logits = torch.tensor([[5., -5, -5], [-5, 5, 5], [5, 5, -5], [-5, -5, -5]])
    onehot = multilabel_logits_to_onehot4(logits, 0.5)
    assert onehot.shape == (4, 4) and onehot.argmax(1).tolist() == [0, 3, 1, 0]


def test_combine_losses_forms():
    m, s = torch.tensor(2.0), torch.tensor(1.0)
    assert torch.isclose(combine_losses(m, s, 0.1, "code"), torch.tensor(0.9 * 2 + 0.1))
    assert torch.isclose(combine_losses(m, s, 0.1, "paper"), torch.tensor(2.1))
    assert torch.equal(combine_losses(m, None, 0.1, "code"), m)


def test_pm_head_layout():
    h = PMHead(768, separate=False)
    assert h.ssl_projector is None and h.ssl_classifier.out_features == 2
    hs = PMHead(768, separate=True, projection_dim=256)
    assert [type(m).__name__ for m in hs.ssl_projector] == ["Linear", "LayerNorm", "ReLU", "Linear"]
    assert hs(torch.randn(3, 768)).shape == (3, 2)


# ------------------------------------------------------------------ dataset
@needs("dataset.pt")
@needs_icbhi
def test_dataset_structure_matches_reference(train_ds):
    g = torch.load(GOLDEN / "dataset.pt", weights_only=False)
    m = g["meta"]
    ds = train_ds
    assert len(ds.all_cycles) == m["n_cycles"] and len(ds) == m["dataset_size"]
    assert ds.n_mixed == m["n_mixed"] and ds.n_ssl == m["n_ssl"]
    assert ds.target_samples == m["target_samples"] and ds.half_target_samples == m["half_target_samples"]
    assert [len(c) for c in ds.classwise_cycle_list] == m["class_counts"]
    assert [len(c) for c in ds.short_classwise_cycle_list] == m["short_counts"]
    assert len(ds.patient_cycles) == m["n_patients"] and list(ds.patient_cycles.keys()) == m["patient_ids"]
    assert [len(v) for v in ds.patient_cycles.values()] == m["patient_sizes"]
    assert [c["filename"] for c in ds.all_cycles] == m["cycle_filenames"]
    assert [c["original_label"] for c in ds.all_cycles] == m["cycle_labels"]
    assert [c["length"] for c in ds.all_cycles] == m["cycle_lengths"]
    assert [list(c) for c in ds.mixing_combinations] == m["combinations"]
    for i, raw in enumerate(g["raw_cycles"]):  # original-length cycles (fade 1/16 + truncation)
        assert torch.equal(ds.all_cycles[i]["audio"], raw), i


@needs("dataset.pt")
@needs_icbhi
def test_dataset_sample_sequence_matches_reference(train_ds):
    """With the same seed and index sequence, the same concatenation/PM samples as the original (python random) come out."""
    g = torch.load(GOLDEN / "dataset.pt", weights_only=False)
    ds = train_ds
    ds.reseed(g["meta"]["seed"])
    n_orig = n_mixed = n_ssl = 0
    for s in g["samples"]:
        x, y, ssl = ds[s["index"]]
        assert x.shape == (160000,), s["index"]
        assert y.tolist() == s["label"], (s["index"], y, s["label"])
        assert int(ssl) == s["ssl_label"], s["index"]
        assert torch.equal(x[:16000], s["head"]), s["index"]
        assert abs(float(x.double().sum()) - s["sum"]) < 1e-3 and abs(float(x.double().abs().sum()) - s["abssum"]) < 1e-3, s["index"]
        if "wav" in s:
            assert torch.equal(x, s["wav"]), s["index"]
        n_orig += s["group"] == "orig"; n_mixed += s["group"] == "mixed"; n_ssl += s["group"] == "ssl"
    assert n_orig >= 6 and n_mixed >= 30 and n_ssl >= 30
    ssl_labels = [s["ssl_label"] for s in g["samples"] if s["group"] == "ssl"]
    assert set(ssl_labels) <= {0, 1} and 0 in ssl_labels and 1 in ssl_labels


@needs("dataset.pt")
@needs_icbhi
def test_test_split_cut_pad_matches_reference():
    """Evaluation path: IndexDataset(fade) + repeat_fade 10 s == original test-cycle cut_pad."""
    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    g = torch.load(GOLDEN / "dataset.pt", weights_only=False)
    pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=10.0, pad_type="repeat_fade"))
    ds = IndexDataset(index_csv=ICBHI, data_root=REPO / ".datasets/icbhi", sampling_rate=16000, sample_level="cycle",
                      split="test", split_file=SPLIT, audio_preprocessor=pre, fade=True)
    assert len(ds) == g["meta"]["test_n_cycles"]
    stems = [pathlib.Path(p).stem for p in ds.df["recording_path"]]
    assert stems[:20] == g["meta"]["test_cycle_filenames_head"]     # index order == original sort order
    for t in g["test_samples"]:
        item = ds[t["index"]]
        assert item["label"] == t["original_label"] and stems[t["index"]] == t["filename"]
        assert torch.equal(item["audio"].squeeze(0), t["wav"]), t["index"]


@needs("dataset.pt")
@needs_icbhi
def test_paper_intent_flags_change_pairing():
    from rsc_toolkit.datasets.pcmcl_concat import PCMCLConcatDataset

    kw = dict(index_csv=ICBHI, data_root=REPO / ".datasets/icbhi", split_file=SPLIT, split="train", mixing_prob=0.0,
              enable_ssl=True, ssl_prob=0.3, rng_seed=1, cache_dir=REPO / "outputs/pcmcl/cache")
    quirk = PCMCLConcatDataset(patient_id_mode="recording", hard_negative_mode="dict_order", **kw)
    paper = PCMCLConcatDataset(patient_id_mode="patient", hard_negative_mode="random_profile", **kw)
    assert len(quirk.patient_cycles) > len(paper.patient_cycles)          # recording (539) vs patient level
    assert all(len(k.split("_")) == 5 for k in quirk.patient_cycles) and all("_" not in k for k in paper.patient_cycles)
    n = len(quirk.all_cycles)
    infos_q = [quirk.get_sample(i) for i in range(n, n + 60)]
    negs_q = [d["sample"]["mix_info"] for d in infos_q if d["ssl_label"] == 0]
    assert negs_q and all(m == "hard_negative_normal_only" for m in negs_q)   # quirk: only the first dict group
    infos_p = [paper.get_sample(i) for i in range(n, n + 200)]
    negs_p = {d["sample"]["mix_info"] for d in infos_p if d["ssl_label"] == 0}
    assert len(negs_p) > 1                                                    # random profile
    basic = PCMCLConcatDataset(hard_negative_mode="basic", **kw)
    assert all(d["sample"]["mix_info"] == "basic_negative" for d in (basic.get_sample(i) for i in range(n, n + 40)) if d["ssl_label"] == 0)


@needs_icbhi
def test_ablation_modes(train_ds):
    from rsc_toolkit.datasets.pcmcl_concat import PCMCLConcatDataset

    kw = dict(index_csv=ICBHI, data_root=REPO / ".datasets/icbhi", split_file=SPLIT, split="train", rng_seed=1,
              cache_dir=REPO / "outputs/pcmcl/cache")
    multi = PCMCLConcatDataset(mode="noconcat_multi", **kw)
    assert len(multi) == len(multi.all_cycles) and multi.n_mixed == 0 and multi.n_ssl == 0
    single = PCMCLConcatDataset(mode="concat_single", mixing_prob=1.0, **kw)
    assert len(single) == 2 * len(single.all_cycles) and single.n_ssl == 0
    x, y, ssl = single[len(single.all_cycles)]
    assert y.dtype == torch.long and int(y) in (0, 1, 2, 3) and int(ssl) == -1
    assert [len(c) for c in single.classwise_cycle_list] == [2063, 1215, 501, 363]  # official train 4-class distribution (dominant label)


@needs("fbank_input.pt")
def test_fbank_matches_reference():
    from rsc_toolkit.processors.mel import generate_fbank

    g = torch.load(GOLDEN / "fbank_input.pt", weights_only=False)
    for w, fb in zip(g["wav"], g["fbank"]):
        ours = generate_fbank(w.unsqueeze(0), sampling_rate=16000, n_mels=128)
        assert ours.shape == (998, 128) and torch.allclose(ours, fb, atol=1e-5)


@needs("specaug.pt")
def test_specaugment_batch_matches_reference():
    from rsc_toolkit.augmentations.specaugment import SpecAugment

    g = torch.load(GOLDEN / "specaug.pt", weights_only=False)
    aug = SpecAugment(policy=g["policy"], mask=g["mask"])
    random.seed(g["seed"]); np.random.seed(g["seed"]); torch.manual_seed(g["seed"])
    x = g["input"].squeeze(1).transpose(1, 2)                 # (B, F, T)
    y = aug(x.clone()).transpose(1, 2).unsqueeze(1)
    assert torch.equal(y, g["output"])


@needs("conversion.pt")
def test_conversion_and_score_match_reference():
    from rsc_toolkit.metrics.icbhi.score import get_score

    g = torch.load(GOLDEN / "conversion.pt", weights_only=False)
    assert multilabel_binary_to_label4(torch.tensor(g["combos"])).tolist() == g["single"]
    sp, se, sc = get_score(g["hits"], g["counts"])
    assert abs(sp - g["sp"]) < 1e-9 and abs(se - g["se"]) < 1e-9 and abs(sc - g["score"]) < 1e-9


# ------------------------------------------------------------------ model
def _check_heads_and_loss(m, g, feats_ours, atol):
    for p in m.classifier.parameters():
        _det_fill(p)
    for p in m.pm_head.parameters():
        _det_fill(p)
    with torch.no_grad():
        logits3 = m.classifier(feats_ours)
        ssl_logits = m.pm_head(feats_ours)
    assert torch.allclose(logits3, g["logits3"], atol=atol), (logits3 - g["logits3"]).abs().max()
    assert torch.allclose(ssl_logits, g["ssl_logits"], atol=atol)
    # loss: (1-w)·BCE + w·CE(mask) — same as the training path (the golden uses eval features, so compare in eval with dropout off; the PM loss depends on whether meta_labels is given)
    m.eval()
    m.ssl_weight, m.loss_combine = 0.1, "code"
    with torch.no_grad():
        out, loss = m.training_forward(g_input(g), g["labels3"], nn.BCEWithLogitsLoss(), meta_labels=g["ssl_labels"])
    assert torch.allclose(loss, g["loss_w0.1"], rtol=1e-3, atol=1e-4), (loss, g["loss_w0.1"])
    m.loss_combine = "paper"
    with torch.no_grad():
        _, loss_p = m.training_forward(g_input(g), g["labels3"], nn.BCEWithLogitsLoss(), meta_labels=g["ssl_labels"])
    assert torch.allclose(loss_p, g["loss_paper_a0.1"], rtol=1e-3, atol=1e-4)
    # validation path: 4-class long -> one-hot (B,4) + 3-label BCE
    with torch.no_grad():
        onehot, vloss = m.training_forward(g_input(g), torch.tensor([0, 3]), nn.BCEWithLogitsLoss())
    assert onehot.shape == (2, 4) and set(onehot.sum(1).tolist()) == {1.0}
    ref = nn.BCEWithLogitsLoss()(logits3, torch.tensor([[1., 0, 0], [0, 1, 1]]))
    assert torch.allclose(vloss, ref, rtol=1e-3, atol=1e-4)


def g_input(g):
    return g["wav"] if "wav" in g else g["input"].squeeze(1).transpose(1, 2)   # AST: (B, 128, 998) freq_time


@needs("beats.pt")
@needs_beats
def test_beats_pcmcl_matches_reference():
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    g = torch.load(GOLDEN / "beats.pt", weights_only=False)
    m = BEATsPCMCL(n_cls=3, specaug=False).eval()
    with torch.no_grad():
        feats = m.forward_features(g["wav"])
    assert feats.shape == (2, 768) and torch.allclose(feats, g["features"], atol=1e-5), (feats - g["features"]).abs().max()
    _check_heads_and_loss(m, g, feats, atol=1e-4)
    sep = PMHead(768, separate=True)
    for p in sep.parameters():
        _det_fill(p)
    with torch.no_grad():
        assert torch.allclose(sep(feats), g["ssl_logits_separate"], atol=1e-4)


@needs("ast.pt")
@needs_ast
def test_ast_pcmcl_matches_reference():
    from rsc_toolkit.models.ast_pcmcl import ASTPCMCL

    g = torch.load(GOLDEN / "ast.pt", weights_only=False)
    m = ASTPCMCL(n_cls=3, specaug=False, amp=False).eval()
    assert list(m.ast.v.pos_embed.shape) == g["pos_embed_shape"] == [1, 99 * 12 + 2, 768]
    assert list(m.ast.get_shape(10, 10, 998, 128)) == g["f_dim_t_dim"] == [99, 12]
    assert {k: list(v.shape) for k, v in m.ast.v.state_dict().items() if not k.startswith(("head", "head_dist"))} == \
           {k: v for k, v in g["v_keys"].items() if not k.startswith(("head", "head_dist"))}
    with torch.no_grad():
        feats = m.forward_features(g_input(g))
    assert feats.shape == (2, 768) and torch.allclose(feats, g["features"], rtol=1e-4, atol=1e-3), (feats - g["features"]).abs().max()
    _check_heads_and_loss(m, g, feats, atol=1e-2)


@needs_beats
def test_beats_pcmcl_training_step_shapes():
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    m = BEATsPCMCL(n_cls=3, enable_ssl=True).train()
    wav = torch.randn(3, 32000) * 0.05
    y = torch.tensor([[1., 0, 0], [0, 1, 1], [1, 1, 0]])
    ssl = torch.tensor([-1, 1, 0])
    logits, loss = m.training_forward(wav, y, nn.BCEWithLogitsLoss(), meta_labels=ssl)
    assert logits.shape == (3, 3) and loss.requires_grad
    loss.backward()
    assert m.pm_head.ssl_classifier.weight.grad is not None and m.classifier.weight.grad is not None
    # if all PM labels are -1, BCE only without combination (original ssl_mask.sum()==0)
    _, loss2 = m.training_forward(wav, y, nn.BCEWithLogitsLoss(), meta_labels=torch.tensor([-1, -1, -1]))
    assert torch.isfinite(loss2)
    single = BEATsPCMCL(n_cls=4, label_mode="single", enable_ssl=True)
    assert single.pm_head is None
    lg, ls = single.training_forward(wav, torch.tensor([0, 3, 1]), F.cross_entropy, meta_labels=ssl)
    assert lg.shape == (3, 4)


@needs_icbhi
def test_train_collate_and_adapter_contract(train_ds):
    from rsc_toolkit.datasets.pcmcl_concat import pcmcl_train_collate

    n = len(train_ds.all_cycles)
    batch = pcmcl_train_collate([train_ds[0], train_ds[n], train_ds[len(train_ds) - 1]])
    x, y, ssl = batch
    assert x.shape == (3, 160000) and y.shape == (3, 3) and y.dtype == torch.float32 and ssl.dtype == torch.long
    assert x.squeeze(1).shape == x.shape                                       # PLAdapter batch[0].squeeze(1) is harmless
    assert int(ssl[0]) == -1 and int(ssl[1]) == -1 and int(ssl[2]) in (0, 1)


@pytest.mark.skipif(not pathlib.Path("pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt").exists(), reason="BEATs checkpoint missing")
def test_multilabel_model_rejects_integer_labels_in_training_mode():
    """In training mode, integer labels (misconfigured datamodule concat_single + label_mode=multilabel) do not silently flow into the validation path
    but stop with TypeError. Integer labels in validation mode (one-hot path) are unchanged."""
    import pytest
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    m = BEATsPCMCL(n_cls=3, specaug=False, enable_ssl=False).train()
    wav = torch.randn(2, 16000)
    with pytest.raises(TypeError, match="label_mode=single"):
        m.training_forward(wav, torch.tensor([0, 3]), nn.BCEWithLogitsLoss())
    m.eval()
    with torch.no_grad():
        onehot, _ = m.training_forward(wav, torch.tensor([0, 3]), nn.BCEWithLogitsLoss())
    assert onehot.shape == (2, 4)
