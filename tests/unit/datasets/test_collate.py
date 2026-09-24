"""Five collate implementations -> a single collate_xy_meta. It must reproduce the output (shape, dtype, values) of each former implementation exactly."""
from __future__ import annotations

import pickle

import pytest
import torch

from rsc_toolkit.datasets import collate as C


def _x(i):
    return torch.full((2, 3), float(i))


def test_tuple_collate_matches_former_index_datamodule_behavior():
    # former _tuple_collate: long meta if a 4th element is present, otherwise (x, y)
    x, y, meta = C.tuple_collate([(_x(1), 0, "id1", 2), (_x(2), 3, "id2", 1)])
    assert x.shape == (2, 2, 3) and y.tolist() == [0, 3] and y.dtype == torch.long
    assert meta.tolist() == [2, 1] and meta.dtype == torch.long
    out = C.tuple_collate([(_x(1), 0, "id1"), (_x(2), 3, "id2")])
    assert len(out) == 2 and out[1].dtype == torch.long


def test_qlung_collate_float64_and_optional():
    x, y, aqs = C.qlung_collate([(_x(1), 0, "a", 0.25), (_x(2), 1, "b", 0.5)])
    assert aqs.dtype == torch.float64 and aqs.tolist() == [0.25, 0.5]
    assert len(C.qlung_collate([(_x(1), 0, "a")])) == 2


def test_btscard_sprs_collate_float32_vector_required():
    x, y, meta = C.btscard_sprs_collate([(_x(1), 0, "a", [1, 0, 0, 1, 0, 0, 1]), (_x(2), 2, "b", [0, 1, 0, 0, 1, 0, 0])])
    assert meta.dtype == torch.float32 and meta.shape == (2, 7)
    with pytest.raises(ValueError, match="meta"):
        C.btscard_sprs_collate([(_x(1), 0, "a")])


def test_stack_meta_and_index_meta_collate():
    x, y, meta = C.stack_meta_collate([(_x(1), 0, torch.ones(7)), (_x(2), 1, torch.zeros(7))])
    assert meta.shape == (2, 7) and meta[0].sum() == 7 and y.dtype == torch.long
    x, y, idx = C.index_meta_collate([(_x(1), 0, 5), (_x(2), 1, 9)])
    assert idx.tolist() == [5, 9] and idx.dtype == torch.long


def test_pcmcl_train_collate_stacks_label_tensors():
    y1, y2 = torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])
    x, y, ssl = C.pcmcl_train_collate([(_x(1), y1, torch.tensor(1)), (_x(2), y2, torch.tensor(-1))])
    assert y.shape == (2, 3) and y.dtype == torch.float32 and ssl.tolist() == [1, -1] and ssl.dtype == torch.long
    x, y, ssl = C.pcmcl_train_collate([(_x(1), torch.tensor(2), torch.tensor(0))])
    assert y.dtype == torch.long and y.tolist() == [2]


def test_make_collate_is_picklable_for_dataloader_workers():
    fn = C.make_collate(meta_index=3, meta_dtype=torch.float64)
    fn2 = pickle.loads(pickle.dumps(fn))
    assert fn2([(_x(1), 0, "a", 0.5)])[2].dtype == torch.float64
    with pytest.raises(ValueError):
        C.collate_xy_meta([(_x(1), 0)], label_mode="nope")
    with pytest.raises(ValueError):
        C.collate_xy_meta([(_x(1), 0, 1)], meta_index=2, meta_mode="nope")


def test_none_label_is_a_clear_error():
    import pytest
    from rsc_toolkit.datasets.collate import collate_xy_meta

    batch = [(torch.zeros(2), 0), (torch.zeros(2), None)]
    with pytest.raises(ValueError, match="label None"):
        collate_xy_meta(batch)
