import math
import torch
import pytest

from rsc_toolkit.metrics.icbhi.score import ICBHIScore, get_score


def _assert_close(a, b, tol=1e-6):
    # works with floats or 0-d tensors
    if torch.is_tensor(a):
        a = a.item()
    if torch.is_tensor(b):
        b = b.item()
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def _score_from_metric(metric):
    out = metric.compute()
    return out["S_p"], out["S_e"], out["icbhi_score"]


def test_compute_matches_get_score_manual_state():
    """
    Directly set hits/counts and ensure compute matches get_score exactly.
    """
    metric = ICBHIScore(num_classes=4, normal_index=0, topk=1)

    # Suppose:
    # class0: 8/10 correct, class1: 1/2, class2: 3/4, class3: 0/1
    hits = torch.tensor([8, 1, 3, 0], dtype=torch.long)
    counts = torch.tensor([10, 2, 4, 1], dtype=torch.long)

    metric.hits = hits.clone()
    metric.counts = counts.clone()

    sp_ref, se_ref, sc_ref = get_score(hits.tolist(), counts.tolist())

    sp, se, sc = _score_from_metric(metric)
    _assert_close(sp, sp_ref)
    _assert_close(se, se_ref)
    _assert_close(sc, sc_ref)


def test_update_produces_expected_hits_counts_top1_and_matches_get_score():
    """
    End-to-end: build logits so we know which ones are correct for top-1.
    """
    metric = ICBHIScore(num_classes=4, normal_index=0, topk=1)

    # targets: 0,0,1,2,3,1,2,3 (8 samples)
    y = torch.tensor([0, 0, 1, 2, 3, 1, 2, 3])

    # Construct logits so predicted class is:
    # pred:    0,1,1,2,0,3,2,3
    # correct: T,F,T,T,F,F,T,T
    preds = torch.tensor([0, 1, 1, 2, 0, 3, 2, 3])

    # Create logits: one-hot-ish with a big value at preds
    logits = torch.full((len(y), 4), -10.0)
    logits[torch.arange(len(y)), preds] = 10.0

    metric.update(logits, y)

    # Expected counts per class:
    # class0: 2, class1: 2, class2: 2, class3: 2
    # Expected hits per class (from correct mask above):
    # sample indices per class:
    # 0-class: idx 0(T),1(F) => hits0=1
    # 1-class: idx 2(T),5(F) => hits1=1
    # 2-class: idx 3(T),6(T) => hits2=2
    # 3-class: idx 4(F),7(T) => hits3=1
    expected_counts = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    expected_hits = torch.tensor([1, 1, 2, 1], dtype=torch.long)

    assert torch.equal(metric.counts.cpu(), expected_counts)
    assert torch.equal(metric.hits.cpu(), expected_hits)

    sp_ref, se_ref, sc_ref = get_score(expected_hits.tolist(), expected_counts.tolist())
    sp, se, sc = _score_from_metric(metric)

    _assert_close(sp, sp_ref)
    _assert_close(se, se_ref)
    _assert_close(sc, sc_ref)


def test_multiple_updates_accumulate_and_match_get_score():
    """
    Two update calls should sum hits/counts and match get_score on the totals.
    """
    metric = ICBHIScore(num_classes=4, normal_index=0, topk=1)

    # Batch 1
    y1 = torch.tensor([0, 1, 2, 3])
    preds1 = torch.tensor([0, 1, 0, 3])  # correct: T,T,F,T
    logits1 = torch.full((4, 4), -10.0)
    logits1[torch.arange(4), preds1] = 10.0
    metric.update(logits1, y1)

    # Batch 2
    y2 = torch.tensor([0, 1, 2, 3])
    preds2 = torch.tensor([2, 1, 2, 1])  # correct: F,T,T,F
    logits2 = torch.full((4, 4), -10.0)
    logits2[torch.arange(4), preds2] = 10.0
    metric.update(logits2, y2)

    # Total counts: each class appears twice => [2,2,2,2]
    # Total hits:
    # class0: batch1 idx0 correct (1), batch2 idx0 wrong (0) => 1
    # class1: batch1 correct (1), batch2 correct (1) => 2
    # class2: batch1 wrong (0), batch2 correct (1) => 1
    # class3: batch1 correct (1), batch2 wrong (0) => 1
    expected_counts = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    expected_hits = torch.tensor([1, 2, 1, 1], dtype=torch.long)

    assert torch.equal(metric.counts.cpu(), expected_counts)
    assert torch.equal(metric.hits.cpu(), expected_hits)

    sp_ref, se_ref, sc_ref = get_score(expected_hits.tolist(), expected_counts.tolist())
    sp, se, sc = _score_from_metric(metric)

    _assert_close(sp, sp_ref)
    _assert_close(se, se_ref)
    _assert_close(sc, sc_ref)


def test_topk_changes_hits_and_still_matches_get_score():
    """
    Verify that top-2 gives credit when target is 2nd best, and compute still matches get_score.
    """
    y = torch.tensor([0, 1, 2, 3])

    # We'll make logits where for sample 0 target=0 is top1 (correct),
    # sample 1 target=1 is rank2 (so top1 incorrect, top2 correct),
    # sample 2 target=2 is rank2,
    # sample 3 target=3 is not in top2.
    logits = torch.tensor([
        [10.0,  0.0, -1.0, -2.0],  # y=0 -> top1=0 correct
        [ 9.0,  8.0,  0.0, -1.0],  # y=1 -> top1=0 wrong, top2 includes 1 correct
        [ 9.0,  0.0,  8.0, -1.0],  # y=2 -> top1=0 wrong, top2 includes 2 correct
        [ 9.0,  8.0,  7.0,  0.0],  # y=3 -> top2 are 0,1 (no 3) incorrect
    ])

    m1 = ICBHIScore(num_classes=4, normal_index=0, topk=1)
    m2 = ICBHIScore(num_classes=4, normal_index=0, topk=2)

    m1.update(logits, y)
    m2.update(logits, y)

    # For top1:
    # correct flags: [T, F, F, F]
    # hits per class: class0=1, class1=0, class2=0, class3=0; counts=[1,1,1,1]
    hits1 = torch.tensor([1, 0, 0, 0], dtype=torch.long)
    counts = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    assert torch.equal(m1.hits.cpu(), hits1)
    assert torch.equal(m1.counts.cpu(), counts)

    sp_ref, se_ref, sc_ref = get_score(hits1.tolist(), counts.tolist())
    sp, se, sc = _score_from_metric(m1)
    _assert_close(sp, sp_ref)
    _assert_close(se, se_ref)
    _assert_close(sc, sc_ref)

    # For top2:
    # correct flags: [T, T, T, F]
    # hits per class: class0=1, class1=1, class2=1, class3=0; counts same
    hits2 = torch.tensor([1, 1, 1, 0], dtype=torch.long)

    assert torch.equal(m2.hits.cpu(), hits2)
    assert torch.equal(m2.counts.cpu(), counts)

    sp_ref, se_ref, sc_ref = get_score(hits2.tolist(), counts.tolist())
    sp, se, sc = _score_from_metric(m2)
    _assert_close(sp, sp_ref)
    _assert_close(se, se_ref)
    _assert_close(sc, sc_ref)


def test_update_raises_on_bad_logits_shape():
    metric = ICBHIScore(num_classes=4, normal_index=0, topk=1)
    logits = torch.randn(2, 3, 4)  # bad shape
    y = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        metric.update(logits, y)


def test_update_raises_on_out_of_range_labels():
    """Labels outside [0, num_classes) are not silently dropped (7-class labels with a 4-class metric used to give Score ≈ 100)."""
    from rsc_toolkit.metrics.icbhi.score import ICBHIScore

    m = ICBHIScore(num_classes=4)
    logits = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="out-of-range labels"):
        m.update(logits, torch.tensor([0, 1, 6]))
    with pytest.raises(ValueError):
        m.update(logits, torch.tensor([-1, 0, 0]))
    m.update(logits, torch.tensor([0, 3, 2]))          # boundary values are allowed
    m.update(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))   # empty batch is harmless
