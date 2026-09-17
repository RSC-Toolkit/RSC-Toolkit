"""Single implementation of batch collate functions.

The toolkit batch contract is the positional tuple `(x, y[, meta])` (see rsc_toolkit/datasets/contract.py):
- x: stack of sample tensors
- y: labels -- by default integers -> long tensor (`label_mode="long"`); when samples already carry label tensors
  (e.g. multi-label float for PC-MCL), stack them (`label_mode="stack"`)
- meta: 3rd batch element. Which sample-tuple slot (`meta_index`) is collapsed, how (`meta_mode`: as_tensor|stack) and with which dtype
  differed per paper. `make_collate(...)` builds all variants; the legacy names remain as instances with the
  same behaviour (only module-level functions and partials, since they must be passed to DataLoader workers).

With `meta_optional=True`, a sample tuple that does not reach `meta_index` yields only `(x, y)` (val batches of the index path).
"""
from __future__ import annotations

from functools import partial
from typing import Any, List, Optional, Sequence, Tuple

import torch


def collate_xy_meta(
    batch: Sequence[Sequence[Any]],
    *,
    label_mode: str = "long",
    meta_index: Optional[int] = None,
    meta_mode: str = "as_tensor",
    meta_dtype: Optional[torch.dtype] = torch.long,
    meta_optional: bool = True,
) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    if label_mode == "long":
        labels = [b[1] for b in batch]
        if any(lab is None for lab in labels):
            # By contract label is int|None (ICBHI clip indexes have empty labels) -- if None ends up in a training batch,
            # report the cause instead of an obscure TypeError from torch.as_tensor.
            raise ValueError(
                f"{sum(lab is None for lab in labels)}/{len(labels)} samples in the batch have label None -- "
                "an unlabeled index (clip level, etc.) cannot form a supervised batch. Check the sample_level and split settings."
            )
        y = torch.as_tensor(labels, dtype=torch.long)
    elif label_mode == "stack":
        y = torch.stack([b[1] for b in batch], dim=0)
    else:
        raise ValueError(f"label_mode must be long|stack: {label_mode!r}")
    if meta_index is None:
        return x, y
    if len(batch[0]) <= meta_index:
        if meta_optional:
            return x, y
        raise ValueError(f"sample tuple has no meta slot (index {meta_index}): len={len(batch[0])}")
    items = [b[meta_index] for b in batch]
    if meta_mode == "as_tensor":
        meta = torch.as_tensor(items, dtype=meta_dtype)
    elif meta_mode == "stack":
        meta = torch.stack(items, dim=0)
    else:
        raise ValueError(f"meta_mode must be as_tensor|stack: {meta_mode!r}")
    return x, y, meta


def make_collate(**kwargs):
    """`collate_xy_meta` with fixed settings (a picklable partial usable as a DataLoader collate_fn)."""
    return partial(collate_xy_meta, **kwargs)


# ── legacy names (same behaviour) ─────────────────────────────────────────────────────────────
# IndexDataModule default: 4th element = int meta label (sgscl family); (x, y) if absent
tuple_collate = make_collate(meta_index=3, meta_mode="as_tensor", meta_dtype=torch.long)
# QLung: 4th element = AQS python float -> float64 (same dtype as the official default_collate; float64 from loss.py:60 on)
qlung_collate = make_collate(meta_index=3, meta_mode="as_tensor", meta_dtype=torch.float64)
# BTS-CARD SPRSound: 4th element = meta7 vector -> (B, 7) float32
btscard_sprs_collate = make_collate(meta_index=3, meta_mode="as_tensor", meta_dtype=torch.float32, meta_optional=False)
# BTS tensors released by the authors / SPRSound OOD: 3rd element = meta7 tensor -> stack
stack_meta_collate = make_collate(meta_index=2, meta_mode="stack", meta_optional=False)
# RespireNet: 3rd element = chunk index int -> long
index_meta_collate = make_collate(meta_index=2, meta_mode="as_tensor", meta_dtype=torch.long, meta_optional=False)
# PC-MCL training: y is already a tensor (3-label float | 4-class long), 3rd element = ssl_label tensor
pcmcl_train_collate = make_collate(label_mode="stack", meta_index=2, meta_mode="stack", meta_optional=False)

__all__ = [
    "collate_xy_meta", "make_collate", "tuple_collate", "qlung_collate", "btscard_sprs_collate",
    "stack_meta_collate", "index_meta_collate", "pcmcl_train_collate",
]
