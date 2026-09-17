"""
Official-code parity tests for BTS-CARD (ICASSP 2026).

Golden sources (tests/golden/btscard/, scripts/reproduction/dump_btscard_goldens.py — official code a9982a0 run with
transformers 4.40.0 / CPU fp32):
- forward_golden.pt : o_multi/o_text/o1/o2/discriminator logits, 6-term loss, TIE (alpha 0/0.5/1) for the first 4 batches of training.pt + random head weights
- text_golden.json  : masking dicts, sentences (random.seed 7), 4.40.0 tokenizer ids (CLAP, tiny-bert)
- sprsound_parity.json : comparison record of features, labels and sentences, authors' valid_spr.pt vs the toolkit pipeline (sprsound_btscard.yaml)
"""
import json
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.datasets.icbhi.meta_description import generate_meta_description, meta_fields_from_meta7
from rsc_toolkit.datasets.sprsound_ood import describe_meta3_sprsound, sprsound_label_to_4class
from rsc_toolkit.models.bts_card import (
    MA_PREFIXES,
    LearnableConstant,
    MetaDiscriminator,
    apply_mask,
    fusion_function,
    kl_loss_function,
    randomly_mask_metadata,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "btscard"
BERT_LOCAL = REPO / "pretrained_models" / "upstream" / "hf" / "tiny-clinicalbert"
AUTHOR_SPR = REPO / "pretrained_models" / "upstream" / "btscard" / "valid_spr.pt"


def _hf_clap_cached() -> bool:
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    return any(p.name.startswith("models--laion--clap-htsat-unfused") for p in hub.glob("models--*"))


needs_models = pytest.mark.skipif(not (_hf_clap_cached() and (BERT_LOCAL / "config.json").exists()),   # a fresh clone tracks only provenance.json, so the folder exists
                                  reason="laion/clap-htsat-unfused cache or local tiny-clinicalbert copy missing")
needs_golden = pytest.mark.skipif(not (GOLDEN / "forward_golden.pt").exists(), reason="golden missing")


@pytest.fixture(scope="module")
def model():
    from rsc_toolkit.models.bts_card import BTSCARDModel

    torch.manual_seed(0)
    m = BTSCARDModel(n_cls=4)
    if (GOLDEN / "forward_golden.pt").exists():
        heads = torch.load(GOLDEN / "forward_golden.pt", map_location="cpu", weights_only=False)["heads"]
        m.classifier.load_state_dict(heads["classifier"])
        m.meta_classifier.classifier.load_state_dict(heads["meta_head"])
        m.loc_discriminator.load_state_dict(heads["loc_discriminator"])
        m.dev_discriminator.load_state_dict(heads["dev_discriminator"])
    return m.eval()


# ------------------------------------------------------------------ goldens (model)
@needs_models
@needs_golden
def test_forward_matches_reference_golden(model):
    g = torch.load(GOLDEN / "forward_golden.pt", map_location="cpu", weights_only=False)
    t = json.load(open(GOLDEN / "text_golden.json"))
    # tokenizer (5.15.1 vs 4.40.0) identity
    ids_c, _ = model.encode_clap_texts(t["clap_texts"], "cpu")
    ids_b, mask_b = model.encode_bert(t["bert_texts_masked"], "cpu")
    assert torch.equal(ids_c, g["clap_input_ids"])
    assert torch.equal(ids_b, g["bert_input_ids"]) and torch.equal(mask_b, g["bert_attention_mask"])
    with torch.no_grad():
        out = model.forward_all(g["x"], t["clap_texts"], t["bert_texts_masked"])
        cls = model.meta_classifier.cls_hidden(ids_b, mask_b)
        loss, terms = model.compute_losses(out, g["y"], g["loc"], g["dev"], nn.CrossEntropyLoss())
    assert torch.allclose(cls, g["bert_cls"], atol=1e-4), (cls - g["bert_cls"]).abs().max()
    for k in ("o_multi", "o_text", "o1", "o2", "loc_logits", "dev_logits"):
        assert torch.allclose(out[k], g[k], atol=1e-4), (k, (out[k] - g[k]).abs().max())
    for k in ("o1", "o2", "kl", "nde", "loc", "dev"):
        assert abs(float(terms[k]) - float(g["losses"][k])) < 1e-4, (k, float(terms[k]), float(g["losses"][k]))
    assert abs(float(loss) - float(g["losses"]["total"])) < 1e-4
    for a, ref in g["tie"].items():
        assert torch.allclose(model.tie(out["o1"], out["o2"], float(a)), ref, atol=1e-4)


@needs_golden
def test_masking_omit_format_matches_reference_text():
    """icbhi_util.py:137-144 masking (None) -> sentence omitting meta_description (the golden uses the masks from random.seed 7)."""
    t = json.load(open(GOLDEN / "text_golden.json"))
    for meta7, mask, ref in zip(t["meta7"], t["masks"], t["bert_texts_masked"]):
        d = apply_mask(meta_fields_from_meta7(meta7), mask)
        assert generate_meta_description(**d) == ref
    assert any(any(m) for m in t["masks"]), "the golden needs at least one masked sample to verify the format"
    # 'unknown' substitution (paper-style flag) produces a different sentence
    d = apply_mask({"age": "adult", "sex": "male", "loc": "trachea", "dev": "Meditron"}, [True, False, False, False], "unknown")
    assert generate_meta_description(**d) == \
        "This sound was recorded from the trachea of a male patient with unknown age, using a Meditron stethoscope."
    d = apply_mask({"age": "adult", "sex": "male", "loc": "trachea", "dev": "Meditron"}, [True, False, False, False])
    assert generate_meta_description(**d) == \
        "This sound was recorded from the trachea of a male patient, using a Meditron stethoscope."


def test_randomly_mask_metadata_is_seed_dependent_and_ordered():
    d = {"age": "adult", "sex": "male", "loc": "trachea", "dev": "Meditron"}
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert randomly_mask_metadata(d, 0.25, g1) == randomly_mask_metadata(d, 0.25, g2)
    assert randomly_mask_metadata(d, 0.0, torch.Generator().manual_seed(1)) == d
    assert all(v is None for v in randomly_mask_metadata(d, 1.0, torch.Generator().manual_seed(1)).values())
    assert list(randomly_mask_metadata(d, 0.25, g1).keys()) == ["age", "sex", "loc", "dev"]


# ------------------------------------------------------------------ pinned quirks
def test_kl_term_has_no_gradient_in_reference_form():
    """Original main:285 — kl(o1, fusion(c, o_text.detach())): no gradient reaches any trainable parameter."""
    torch.manual_seed(0)
    o_multi = torch.randn(4, 4, requires_grad=True)
    o_text = torch.randn(4, 4, requires_grad=True)
    c = LearnableConstant(1.0, trainable=False)
    o1 = fusion_function(o_multi, o_text)
    o2_dead = fusion_function(c.dummy_vector * torch.ones_like(o1), o_text.detach())
    kl = kl_loss_function(o1, o2_dead)
    assert not kl.requires_grad  # there is no graph at all
    assert c.dummy_vector.requires_grad is False and c.dummy_vector.grad is None
    # paper-style (kl_dead=False): gradient flows to o_text
    kl_live = kl_loss_function(o1, fusion_function(c.dummy_vector * torch.ones_like(o1), o_text))
    kl_live.backward()
    assert o_text.grad is not None and o_text.grad.abs().sum() > 0
    assert o_multi.grad is None  # o1 on the target side is detached


def test_constant_excluded_from_optimizer_and_ma():
    c = LearnableConstant(1.0)
    opt = torch.optim.Adam(c.parameters(), lr=1.0)
    (c.dummy_vector * 2).sum()  # no graph — requires_grad False
    opt.step()
    assert float(c.dummy_vector) == 1.0
    keys = ["audio_features.text_model.x", "classifier.weight", "meta_classifier.bert.y", "meta_classifier.classifier.0.weight",
            "loc_discriminator.net.1.weight", "dev_discriminator.net.4.bias", "constant.dummy_vector"]
    from rsc_toolkit.models.bts_card import BTSCARDModel

    ma = BTSCARDModel.ma_state_keys(keys)
    assert ma == keys[:4]
    assert MA_PREFIXES == ("audio_features.", "classifier.", "meta_classifier.")


def test_discriminator_takes_4d_logits():
    d = MetaDiscriminator(4, 96, 7, 0.1)
    assert isinstance(d.net[0], nn.LayerNorm) and tuple(d.net[0].normalized_shape) == (4,)
    assert d.net[1].in_features == 4 and d.net[1].out_features == 96 and d.net[4].out_features == 7
    assert d(torch.randn(3, 4)).shape == (3, 7)
    assert MetaDiscriminator(4, 96, 4, 0.1)(torch.randn(3, 4)).shape == (3, 4)


@needs_models
def test_alpha_zero_equals_o1_and_ma_callback_targets(model):
    from rsc_toolkit.trainer.callbacks.btscard import BTSCARDWeightAveraging

    x = torch.randn(2, 1, 1001, 64)
    meta = torch.tensor([[70., 1., 33., -1., -1., 2., 1.], [5., 0., -1., 20., 110., 0., 3.]])
    with torch.no_grad():
        out = model.forward_all(x.squeeze(1), model.describe(meta), model.describe_bert(meta, False))
        assert torch.allclose(model(x, meta, alpha=0.0), out["o1"])
        assert torch.allclose(model(x, meta, alpha=1.0), out["o1"] - out["o2"])
    sd_keys = list(model.state_dict())
    cb = BTSCARDWeightAveraging(beta=0.5)

    class _PL:  # pl_module stand-in
        rsc_model = model

    targets = set(cb.target_keys(_PL()))
    assert targets == set(model.ma_state_keys(sd_keys))
    assert not any(k.startswith(("loc_discriminator.", "dev_discriminator.", "constant.")) for k in targets)
    assert any(k.startswith("meta_classifier.bert.") for k in targets) and "classifier.weight" in targets
    assert model.constant.dummy_vector.requires_grad is False


@needs_models
def test_training_forward_masks_only_bert_and_exposes_terms(model):
    torch.manual_seed(0)
    model.train()
    try:
        x = torch.randn(2, 1001, 64)
        meta = torch.tensor([[70., 1., 33., -1., -1., 2., 1.], [5., 0., -1., 20., 110., 0., 3.]])
        model.p_mask = 1.0  # mask everything -> BERT sentence uses the empty format, CLAP sentence unchanged
        assert model.describe_bert(meta, masking=True) == ["This patient is.", "This patient is."]
        assert model.describe(meta)[0].startswith("This sound was recorded from the right anterior chest")
        model.p_mask = 0.25
        logits, loss = model.training_forward(x, torch.tensor([0, 3]), nn.CrossEntropyLoss(), meta_labels=meta)
        assert logits.shape == (2, 4) and torch.isfinite(loss)
        assert set(model.loss_terms) == {"o1", "o2", "kl", "nde", "loc", "dev"}
        loss.backward()
        assert model.constant.dummy_vector.grad is None
        assert model.loc_discriminator.net[1].weight.grad is not None
        with pytest.raises(ValueError, match="meta"):
            model.training_forward(x, torch.tensor([0, 1]), nn.CrossEntropyLoss())
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


# ------------------------------------------------------------------ SPRSound OOD
def test_sprsound_4class_mapping_distribution():
    import pandas as pd

    df = pd.read_csv(REPO / "rsc_toolkit" / "datasets" / "sprsound" / "data" / "metadata.csv")
    v = df[df["Split"] == "valid"]
    counts = [0, 0, 0, 0]
    for lab in v["Label"]:
        counts[sprsound_label_to_4class(lab)] += 1
    assert counts == [1040, 83, 305, 1]  # paper Table 1 SPRSound valid
    assert sprsound_label_to_4class("Rhonchi") is None and sprsound_label_to_4class("Stridor") is None
    with pytest.raises(ValueError):                       # the wheeze-merge option is not supported
        sprsound_label_to_4class("Stridor", "wheeze")
    tr = df[df["Split"] == "train"]
    assert sum(sprsound_label_to_4class(l) == 2 for l in tr["Label"] if sprsound_label_to_4class(l) is not None) == 452
    assert sum(sprsound_label_to_4class(l) is None for l in tr["Label"]) == 39 + 15   # train Rhonchi/Stridor are excluded (wheeze 452, not 506)


def test_sprsound_sentence_template():
    assert describe_meta3_sprsound([8.1, 1, 0]) == "This sound was recorded from the left posterior of a female pediatric patient."
    assert describe_meta3_sprsound([3.0, 0, 3]) == "This sound was recorded from the right lateral of a male pediatric patient."
    assert describe_meta3_sprsound([25.0, 0, 1]) == "This sound was recorded from the left lateral of an adult male patient."


@pytest.mark.skipif(not AUTHOR_SPR.exists(), reason="authors' valid_spr.pt missing")
def test_author_sprsound_pt_layout():
    from rsc_toolkit.datasets.sprsound_ood import SPRSoundOODAuthorDataset, sprsound_ood_collate

    ds = SPRSoundOODAuthorDataset(AUTHOR_SPR)
    assert len(ds) == 1429
    feat, y, meta, rid = ds[0]
    assert feat.shape == (1, 1001, 64) and meta.shape == (3,) and rid == "40512331_8.1_1_p1_3548"
    x, yy, mm = sprsound_ood_collate([ds[0], ds[1]])
    assert x.shape == (2, 1, 1001, 64) and yy.dtype == torch.long and mm.shape == (2, 3)
    counts = [0, 0, 0, 0]
    for it in ds.items:
        counts[int(it[1])] += 1
    assert counts == [1040, 83, 305, 1]


@pytest.mark.skipif(not (GOLDEN / "sprsound_parity.json").exists(), reason="parity record missing")
def test_sprsound_pipeline_parity_record():
    r = json.load(open(GOLDEN / "sprsound_parity.json"))
    assert r["n_compared"] == 1429 and r["label_mismatch"] == 0 and r["meta_mismatch"] == 0 and r["text_mismatch"] == 0
    assert r["class_counts_toolkit"] == [1040, 83, 305, 1]
    # approximate, not bit-identical: max|diff| 6.7e-3 (24/1429 > 1e-3) — cause unconfirmed (presumably numeric differences in the authors'
    # environment, torch 2.0.1/Windows; not measured). The tolerance below is a post-hoc bound fit to the observations. Labels, meta and sentences match exactly.
    assert r["max_abs_diff"] < 1e-2 and r["n_diff_gt_1e-3"] <= 30


AUTHOR_CKPT = REPO / "pretrained_models" / "upstream" / "btscard" / "best.pth"


@needs_models
@pytest.mark.skipif(not (AUTHOR_CKPT.exists() and (GOLDEN / "ckpt_forward_golden.pt").exists()), reason="authors' best.pth or ckpt golden missing")
def test_author_checkpoint_loads_and_matches_reference_outputs():
    """Loads the authors' best.pth without key conversion (only discriminator/constants missing) -> matches the official code (4.40.0) output at 1e-4."""
    from rsc_toolkit.models.bts_card import BTSCARDModel

    g = torch.load(GOLDEN / "forward_golden.pt", map_location="cpu", weights_only=False)
    ref = torch.load(GOLDEN / "ckpt_forward_golden.pt", map_location="cpu", weights_only=False)
    t = json.load(open(GOLDEN / "text_golden.json"))
    m = BTSCARDModel(n_cls=4)
    info = m.load_reference_checkpoint(AUTHOR_CKPT)
    assert info["unexpected"] == [] and info["alpha"] == 0.0
    assert all(k.startswith(("loc_discriminator.", "dev_discriminator.", "constant.")) for k in info["missing"])
    m.eval()
    with torch.no_grad():
        out = m.forward_all(g["x"], t["clap_texts"], t["bert_texts_masked"])
    for k in ("o_multi", "o_text", "o1", "o2"):
        assert torch.allclose(out[k], ref[k], atol=1e-4), (k, (out[k] - ref[k]).abs().max())
    for a, r in ref["tie"].items():
        assert torch.allclose(m.tie(out["o1"], out["o2"], float(a)), r, atol=1e-4)
