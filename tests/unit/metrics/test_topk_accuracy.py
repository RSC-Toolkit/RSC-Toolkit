import math
import torch
import pytest

from rsc_toolkit.metrics.topk_accuracy import TopKAccuracy


def assert_close(a, b, tol=1e-6):
    if torch.is_tensor(a):
        a = a.item()
    if torch.is_tensor(b):
        b = b.item()
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def test_top1_perfect_accuracy():
    metric = TopKAccuracy(topk=(1,))

    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    targets = torch.tensor([0, 1, 2])

    metric.update(logits, targets)
    out = metric.compute()

    assert_close(out["acc@1"], 100.0)


def test_top1_partial_accuracy():
    metric = TopKAccuracy(topk=(1,))

    logits = torch.tensor([
        [10.0, 0.0, 0.0],  # correct
        [10.0, 0.0, 0.0],  # wrong
        [0.0, 0.0, 10.0],  # correct
    ])
    targets = torch.tensor([0, 1, 2])

    metric.update(logits, targets)
    out = metric.compute()

    # 2 / 3 correct
    assert_close(out["acc@1"], 2 / 3 * 100)


def test_topk_behavior():
    metric = TopKAccuracy(topk=(1, 2))

    logits = torch.tensor([
        [10.0, 9.0, 0.0],   # target=1 → correct only in top-2
        [0.0, 10.0, 9.0],   # target=2 → correct only in top-2
    ])
    targets = torch.tensor([1, 2])

    metric.update(logits, targets)
    out = metric.compute()

    assert_close(out["acc@1"], 0.0)
    assert_close(out["acc@2"], 100.0)


def test_multiple_updates_accumulate():
    metric = TopKAccuracy(topk=(1,))

    # batch 1
    logits1 = torch.tensor([
        [10.0, 0.0],
        [0.0, 10.0],
    ])
    targets1 = torch.tensor([0, 1])
    metric.update(logits1, targets1)

    # batch 2 (one correct, one wrong)
    logits2 = torch.tensor([
        [10.0, 0.0],
        [10.0, 0.0],
    ])
    targets2 = torch.tensor([0, 1])
    metric.update(logits2, targets2)

    out = metric.compute()

    # 3 correct / 4 total
    assert_close(out["acc@1"], 75.0)


def test_logits_shape_validation():
    metric = TopKAccuracy(topk=(1,))

    logits = torch.randn(2, 3, 4)  # invalid shape
    targets = torch.tensor([0, 1])

    with pytest.raises(ValueError):
        metric.update(logits, targets)
