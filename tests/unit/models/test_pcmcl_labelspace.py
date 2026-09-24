"""
PC-MCL label-space tests — for ICBHI they **pin equality with the authors' behavior**, for SPRSound they **pin the convention we defined**.

The most important ones are `test_icbhi_*`: they nail down that adding the label-space abstraction did not change a single bit of the ICBHI path.
(Run-level comparison is covered by the golden tests in `test_pcmcl.py`.)
"""
import pathlib
import pytest
import torch

from rsc_toolkit.methods.pcmcl import multilabel_binary_to_label4
from rsc_toolkit.methods.pcmcl_labelspace import ICBHI4, SPRSOUND7, get_label_space

# `pcmcl_concat` pulls in lightning at module top level (pcmcl_concat.py:523). If that is loaded at collection time,
# global state changes earlier than usual, so it is imported **inside functions**, following the same convention as test_pcmcl.py.


def _author_fns():
    from rsc_toolkit.datasets.pcmcl_concat import convert_to_multilabel, multilabel_mixing_combinations

    return convert_to_multilabel, multilabel_mixing_combinations




# ====================================================================== ICBHI regression (authors' behavior pinned)
def test_icbhi_encode_matches_author_function():
    """LabelSpace.to_multilabel == original _convert_to_multilabel."""
    convert_to_multilabel, _ = _author_fns()
    for y in range(4):
        assert ICBHI4.to_multilabel(y) == convert_to_multilabel(y), y


def test_icbhi_decode_matches_author_conversion():
    """LabelSpace.decode_binary == conversion.py convert_original(additive), all 8 combinations."""
    combos = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    assert ICBHI4.decode_binary(combos).tolist() == multilabel_binary_to_label4(combos).tolist()
    assert ICBHI4.decode_binary(combos).tolist() == [0, 2, 1, 3, 0, 2, 1, 3]


def test_icbhi_buckets_match_author_rule():
    """Inclusion (⊇) rule == the original's '0, 1, 2 are bits, 3 is original_label==3'. All 4 classes."""
    convert_to_multilabel, _ = _author_fns()
    for y in range(4):
        ml = convert_to_multilabel(y)
        author = [k for k in range(3) if ml[k] > 0] + ([3] if y == 3 else [])
        assert ICBHI4.buckets_for(ml) == sorted(author), (y, ml)


def test_icbhi_combinations_unchanged():
    _, multilabel_mixing_combinations = _author_fns()
    for strategy in ("additive", "dominant"):
        assert ICBHI4.mixing_combinations(strategy) == multilabel_mixing_combinations(strategy)


def test_icbhi_pairs_are_union_of_attributes():
    """Demonstrates that the authors' additive list is 'all unordered pairs + union labels' — the basis for the SPRSound extension."""
    combos = ICBHI4.mixing_combinations("additive")
    pairs = [c for c in combos if ICBHI4.sources_in(c) == 2]
    assert len(pairs) == 10                                    # C(4+1, 2)
    assert {(c[0], c[1]) if c[0] <= c[1] else (c[1], c[0]) for c in pairs} == \
           {(i, j) for i in range(4) for j in range(i, 4)}
    for c in pairs:
        union = [max(a, b) for a, b in zip(ICBHI4.to_multilabel(c[0]), ICBHI4.to_multilabel(c[1]))]
        assert ICBHI4.label_in(c) == union, c
    triples = [c for c in combos if ICBHI4.sources_in(c) == 3]
    assert len(triples) == 1 and (triples[0][0], triples[0][1], triples[0][2]) == (0, 1, 2)


def test_sources_in_reproduces_author_length_branches():
    """The original branched on length 6/7 for the number of sources — recomputation gives the same answer."""
    for c in ICBHI4.mixing_combinations("additive"):
        assert ICBHI4.sources_in(c) == (2 if len(c) == 6 else 3)


# ====================================================================== SPRSound (toolkit convention pinned)
def test_sprsound_decomposition_is_lossless():
    """The 7 classes have distinct attribute vectors, and decoding recovers the original class."""
    vecs = [tuple(SPRSOUND7.to_multilabel(y)) for y in range(7)]
    assert len(set(vecs)) == 7
    bits = torch.tensor(vecs)
    assert SPRSOUND7.decode_binary(bits).tolist() == list(range(7))


def test_sprsound_attribute_sharing_is_real():
    """Why MCL works — one attribute is active in several classes."""
    discont = [y for y in range(7) if SPRSOUND7.to_multilabel(y)[1] > 0]
    cont = [y for y in range(7) if SPRSOUND7.to_multilabel(y)[2] > 0]
    assert discont == [1, 3, 4]          # Fine · Coarse · Wheeze+Crackle
    assert cont == [2, 4, 5, 6]          # Wheeze · Wheeze+Crackle · Rhonchi · Stridor


def test_sprsound_axes_mirror_icbhi():
    """Attributes 0, 1, 2 must play the same roles as in ICBHI so the PM pathology-profile code runs unchanged."""
    assert SPRSOUND7.attr_names[:3] == ("normal", "discont", "cont")
    assert SPRSOUND7.profile_attrs == ICBHI4.profile_attrs == (0, 1, 2)
    # Wheeze+Crackle takes the same slot as ICBHI both (discontinuous ∧ continuous)
    assert SPRSOUND7.to_multilabel(4)[1:3] == [1, 1]


def test_sprsound_buckets_pool_by_attribute():
    """Bucket 1 (Fine Crackle) pools the whole discontinuous family — the same structure as both going into the crackle bucket in the authors' code."""
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(3)) == [1, 3]      # Coarse -> discontinuous, itself
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(4)) == [1, 2, 4]   # W+C -> discontinuous, continuous, itself
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(5)) == [2, 5]      # Rhonchi -> continuous, itself
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(0)) == [0]


def test_sprsound_combinations_are_all_pairs_with_union_labels():
    combos = SPRSOUND7.mixing_combinations("additive")
    assert len(combos) == 28                                   # C(7+1, 2)
    assert all(SPRSOUND7.sources_in(c) == 2 for c in combos)   # no 3-source combinations (see module docstring)
    assert {(c[0], c[1]) for c in combos} == {(i, j) for i in range(7) for j in range(i, 7)}
    for c in combos:
        union = [max(a, b) for a, b in zip(SPRSOUND7.to_multilabel(c[0]), SPRSOUND7.to_multilabel(c[1]))]
        assert SPRSOUND7.label_in(c) == union, c


def test_sprsound_decode_priority_is_documented_behaviour():
    """Decoding of attribute combinations that do not occur — **a rule we defined**, so it is pinned here."""
    ls = SPRSOUND7
    # an adventitious sound wins even if the normal bit is on (same structure as ICBHI)
    assert ls.decode_binary(torch.tensor([[1, 1, 0, 0, 0, 0]])).tolist() == [1]
    # no attribute on -> normal
    assert ls.decode_binary(torch.tensor([[0, 0, 0, 0, 0, 0]])).tolist() == [0]
    # discontinuous+continuous comes before anything else — Wheeze+Crackle even if coarse is also on
    assert ls.decode_binary(torch.tensor([[0, 1, 1, 1, 0, 0]])).tolist() == [4]
    # coarse only qualifies discontinuous — coarse alone without discontinuous is ignored
    assert ls.decode_binary(torch.tensor([[0, 0, 1, 1, 0, 0]])).tolist() == [2]
    # if rhonchi and stridor are both on, rhonchi wins (arbitrary choice)
    assert ls.decode_binary(torch.tensor([[0, 0, 1, 0, 1, 1]])).tolist() == [5]


def test_sprsound_rejects_dominant_strategy():
    with pytest.raises(NotImplementedError):
        SPRSOUND7.mixing_combinations("dominant")


def test_out_of_range_label_raises_instead_of_corrupting():
    """The most important guard — feeding 7-class labels to the 4-class converter used to silently turn labels 4, 5, 6 into [0,0,0]."""
    convert_to_multilabel, _ = _author_fns()
    assert convert_to_multilabel(5) == [0, 0, 0]        # old behavior (for reference): silently lost
    with pytest.raises(ValueError, match="class index"):
        ICBHI4.to_multilabel(5)                         # new behavior: fails immediately


def test_get_label_space_rejects_unknown():
    with pytest.raises(ValueError, match="label_space"):
        get_label_space("icbhi7")


# ====================================================================== model wiring
def test_model_requires_matching_n_cls():
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    with pytest.raises(ValueError, match="number of attributes"):
        BEATsPCMCL(n_cls=3, label_space="sprsound7")
    with pytest.raises(ValueError, match="number of attributes"):
        BEATsPCMCL(n_cls=7, label_space="sprsound7")     # 7 is the number of classes, not attributes


@pytest.mark.skipif(not pathlib.Path("pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt").exists(), reason="BEATs checkpoint missing")  # needs_beats_ckpt
def test_pm_head_survives_in_sprsound_multilabel():
    """Key point — the multi-label path is kept, so the PM (patient-consistency) head stays alive."""
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    m = BEATsPCMCL(n_cls=6, label_space="sprsound7", enable_ssl=True)
    assert m.pm_head is not None and m.pm_head.ssl_classifier.out_features == 2
    assert m.classifier.out_features == 6
    # validation path: 7-class long -> (B, 7) one-hot + 6-attribute BCE
    onehot = m.ls.logits_to_onehot(torch.randn(5, 6))
    assert onehot.shape == (5, 7) and set(onehot.sum(1).tolist()) == {1.0}
