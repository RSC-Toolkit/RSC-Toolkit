"""
Official-code parity tests for the BTS group (Audio-CLAP, BTS).

Golden sources:
- tests/golden/bts/meta_descriptions.json — sentences from running the official util/icbhi_util.get_meta_text_descriptions
  on the 28 real test.pt meta combinations x 15 meta_modes (all 206,940 compared, 0 mismatches).
- tests/golden/bts/checkpoint_eval_record.json — evaluation record of loading the authors' best.pth
  (AMP: Sp 82.58 / Se 45.11 / Score 63.85 = README). The file is 1.8 GB, so it is not run here;
  reproduce with scripts/reproduction/eval_bts_checkpoint.py.
- tests/golden/bts/pipeline_parity_test_split.json — comparison record of the toolkit 48 kHz pipeline vs the authors' .pt.
"""
import json
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.datasets.icbhi.meta_description import (
    META_MODES,
    describe_meta7,
    generate_meta_description,
    meta7_from_index_meta,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "bts"
CKPT = REPO / "pretrained_models" / "upstream" / "bts" / "best.pth"


def _hf_clap_cached() -> bool:
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    return any(p.name.startswith("models--laion--clap-htsat-unfused") for p in hub.glob("models--*"))


needs_clap = pytest.mark.skipif(not _hf_clap_cached(), reason="laion/clap-htsat-unfused not in the HF cache")


# ------------------------------------------------------------- meta sentences
@pytest.mark.skipif(not (GOLDEN / "meta_descriptions.json").exists(), reason="golden missing")
def test_meta_descriptions_match_reference_golden():
    golden = json.load(open(GOLDEN / "meta_descriptions.json"))
    assert len(golden) >= 15 * 20
    modes_seen = set()
    for g in golden:
        assert describe_meta7(g["meta7"], g["mode"]) == g["text"], (g["mode"], g["meta7"])
        modes_seen.add(g["mode"])
    assert modes_seen == set(META_MODES)


def test_generate_meta_description_examples():
    """Original sentence format (same pattern as the paper Fig. 2 example)."""
    assert generate_meta_description("adult", "male", "trachea", "Meditron") == \
        "This sound was recorded from the trachea of an adult male patient, using a Meditron stethoscope."
    assert generate_meta_description("pediatric", "female", None, "AKGC417L") == \
        "This sound was recorded from a female pediatric patient, using an AKGC417L microphone."
    assert generate_meta_description("adult", None, None, None) == "This patient is an adult patient."
    assert generate_meta_description(None, None, "left anterior chest", None) == \
        "This sound was recorded from the left anterior chest."


def test_unknown_age_is_pediatric_like_reference():
    """Original quirk: a missing age (-1) is below 19, so it becomes pediatric."""
    assert "pediatric" in describe_meta7([-1, 0, -1, -1, -1, 0, 0], "age")


def test_meta7_from_index_meta_matches_reference_layout():
    meta = {"age": 70.0, "sex": "F", "adult_BMI": 33.0, "child_weight": float("nan"),
            "child_height": None, "chest_location": "Ar", "device": "LittC2SE"}
    assert meta7_from_index_meta(meta) == [70.0, 1.0, 33.0, -1.0, -1.0, 2.0, 1.0]
    assert describe_meta7(meta7_from_index_meta(meta), "all") == \
        "This sound was recorded from the right anterior chest of an adult female patient, using a LittC2SE stethoscope."


# ---------------------------------------------------------- data loader
def test_precomputed_dataset_and_collate(tmp_path):
    from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataset, bts_collate

    items = [(torch.randn(1, 1001, 64), 2, torch.tensor([70., 1., 33., -1., -1., 2., 1.], dtype=torch.float64)),
             (torch.randn(1, 1001, 64), 0, torch.tensor([5., 0., -1., 20., 110., 0., 3.], dtype=torch.float64))]
    p = tmp_path / "fake.pt"
    torch.save(items, p)
    ds = BTSPrecomputedDataset(p)
    x, y, meta = bts_collate([ds[0], ds[1]])
    assert x.shape == (2, 1, 1001, 64) and y.tolist() == [2, 0]
    assert meta.shape == (2, 7) and meta.dtype == torch.float32


# ----------------------------------------------------------------- model
@needs_clap
def test_bts_model_state_dict_matches_reference_checkpoint_layout():
    from rsc_toolkit.models.bts import BTSModel

    m = BTSModel(n_cls=4)
    keys = set(m.state_dict())
    assert {"classifier.weight", "classifier.bias"} <= keys
    clap_keys = {k for k in keys if k.startswith("audio_features.")}
    assert len(clap_keys) == 447  # transformers 5.15.1 ClapModel == number of keys in the authors' ckpt
    assert tuple(m.classifier.weight.shape) == (4, 1024)  # concat: text 512 + audio 512

    if CKPT.exists():
        m.load_reference_checkpoint(CKPT)  # strict=True — passes only if the key sets are exactly equal


@needs_clap
def test_bts_training_forward_uses_text_and_audio():
    from rsc_toolkit.models.bts import BTSModel

    m = BTSModel(n_cls=4).eval()
    x = torch.randn(2, 1, 1001, 64)
    meta = torch.tensor([[70., 1., 33., -1., -1., 2., 1.], [5., 0., -1., 20., 110., 0., 3.]])
    with torch.no_grad():
        logits, loss = m.training_forward(x.squeeze(1), torch.tensor([0, 1]), nn.CrossEntropyLoss(), meta_labels=meta)
        same_audio_other_meta = m(x, meta[[1, 0]])
    assert logits.shape == (2, 4) and torch.isfinite(loss)
    assert not torch.allclose(logits, same_audio_other_meta)  # the meta sentence actually affects the output

    with pytest.raises(ValueError, match="meta"):
        m.training_forward(x, torch.tensor([0, 1]), nn.CrossEntropyLoss())


@needs_clap
def test_audioclap_accepts_3d_and_4d_input():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("audioclap")(n_cls=4).eval()
    x = torch.randn(2, 1, 1001, 64)
    with torch.no_grad():
        a = m(x)
        b = m(x.squeeze(1))
        logits, loss = m.training_forward(x.squeeze(1), torch.tensor([0, 3]), nn.CrossEntropyLoss(),
                                          meta_labels=torch.zeros(2, 7))
    assert torch.allclose(a, b) and logits.shape == (2, 4) and torch.isfinite(loss)
