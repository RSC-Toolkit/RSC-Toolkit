"""Single source of Se/Sp/Score (sesp_score) and ICBHIScore key parameters."""
from __future__ import annotations

import pytest
import torch

from rsc_toolkit.metrics.icbhi.score import ICBHIScore, get_score
from rsc_toolkit.metrics.icbhi.sesp import EPS, sesp_score


def test_sesp_score_matches_get_score_and_icbhi_score_compute():
    hits, counts = [100, 30, 20, 5], [150, 60, 40, 10]
    assert sesp_score(hits, counts) == get_score(hits, counts)
    m = ICBHIScore(num_classes=4)
    m.hits += torch.tensor(hits)
    m.counts += torch.tensor(counts)
    out = m.compute()
    sp, se, sc = sesp_score(hits, counts)
    assert out["S_p"].item() == pytest.approx(sp, abs=1e-4)
    assert out["S_e"].item() == pytest.approx(se, abs=1e-4)
    assert out["icbhi_score"].item() == pytest.approx(sc, abs=1e-4)


def test_sesp_score_normal_index_and_eps():
    hits, counts = [5, 90, 10], [10, 100, 20]
    sp, se, sc = sesp_score(hits, counts, normal_index=1)
    assert sp == pytest.approx(90.0, abs=1e-6) and se == pytest.approx(50.0, abs=1e-6) and sc == pytest.approx(70.0, abs=1e-6)
    sp, se, _ = sesp_score([0, 0], [0, 0])  # empty class gives 0 (eps in denominator)
    assert sp == 0.0 and se == 0.0
    assert EPS == 1e-10
    with pytest.raises(ValueError):
        sesp_score([1], [1, 2])
    with pytest.raises(ValueError):
        sesp_score([1, 2], [1, 2], normal_index=2)


def test_icbhi_score_key_parameters_default_unchanged():
    m = ICBHIScore(num_classes=7, normal_index=0)
    assert set(m.compute()) == {"S_p", "S_e", "icbhi_score"}
    assert m.keys == {"S_p": "S_p", "S_e": "S_e", "score": "icbhi_score"}
    s = ICBHIScore(num_classes=7, score_key="sprsound_score", key_prefix="val/")
    assert set(s.compute()) == {"val/S_p", "val/S_e", "val/sprsound_score"}
    assert s.keys["score"] == "val/sprsound_score"
