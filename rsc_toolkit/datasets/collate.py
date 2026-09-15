"""배치 collate 의 단일 구현 (리팩토링 P2-6).

툴킷의 배치 계약은 위치 튜플 `(x, y[, meta])` 다 (rsc_toolkit/datasets/contract.py 참조):
- x: 샘플 텐서 stack
- y: 라벨 — 기본은 정수 → long 텐서(`label_mode="long"`); PC-MCL 처럼 샘플이 이미 라벨 텐서(멀티라벨 float 등)를
  주면 stack(`label_mode="stack"`)
- meta: 배치 3번째 원소. 샘플 튜플의 어느 자리(`meta_index`)를 어떤 방식(`meta_mode`: as_tensor|stack)·dtype 으로
  접을지가 논문마다 달랐고 그래서 collate 가 5벌 있었다. 이제 `make_collate(...)` 하나로 만들고, 예전 이름들은
  같은 동작의 인스턴스로 남긴다 (DataLoader 워커에 넘어가야 하므로 module-level 함수·partial 만 쓴다).

`meta_optional=True` 면 샘플 튜플이 `meta_index` 까지 없을 때 `(x, y)` 두 원소만 돌려준다 (index 경로의 val 배치).
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
            # 계약상 label 은 int|None(ICBHI clip 인덱스는 label 빈 값) — 학습 배치에 None 이 섞이면 torch.as_tensor 의
            # 알 수 없는 TypeError 대신 원인을 말한다 (REVIEW 2026-09-10 §2 P2).
            raise ValueError(
                f"배치에 label 이 None 인 샘플이 {sum(lab is None for lab in labels)}/{len(labels)} 개 있다 — "
                "라벨 없는 인덱스(clip 단위 등)로는 지도 학습 배치를 만들 수 없다. sample_level·split 설정을 확인하라."
            )
        y = torch.as_tensor(labels, dtype=torch.long)
    elif label_mode == "stack":
        y = torch.stack([b[1] for b in batch], dim=0)
    else:
        raise ValueError(f"label_mode 는 long|stack: {label_mode!r}")
    if meta_index is None:
        return x, y
    if len(batch[0]) <= meta_index:
        if meta_optional:
            return x, y
        raise ValueError(f"샘플 튜플에 meta 자리(index {meta_index})가 없다: len={len(batch[0])}")
    items = [b[meta_index] for b in batch]
    if meta_mode == "as_tensor":
        meta = torch.as_tensor(items, dtype=meta_dtype)
    elif meta_mode == "stack":
        meta = torch.stack(items, dim=0)
    else:
        raise ValueError(f"meta_mode 는 as_tensor|stack: {meta_mode!r}")
    return x, y, meta


def make_collate(**kwargs):
    """`collate_xy_meta` 의 설정 고정 버전 (DataLoader collate_fn 으로 넘길 수 있는 picklable partial)."""
    return partial(collate_xy_meta, **kwargs)


# ── 예전 이름들 (동작 동일) ─────────────────────────────────────────────────────────────
# IndexDataModule 기본: 4번째 원소 = int meta 라벨 (sgscl 계열), 없으면 (x, y)
tuple_collate = make_collate(meta_index=3, meta_mode="as_tensor", meta_dtype=torch.long)
# QLung: 4번째 원소 = AQS python float → float64 (공식 default_collate 와 같은 dtype; loss.py:60 부터 float64)
qlung_collate = make_collate(meta_index=3, meta_mode="as_tensor", meta_dtype=torch.float64)
# BTS-CARD SPRSound: 4번째 원소 = meta7 벡터 → (B, 7) float32
btscard_sprs_collate = make_collate(meta_index=3, meta_mode="as_tensor", meta_dtype=torch.float32, meta_optional=False)
# BTS 저자 배포 텐서 / SPRSound OOD: 3번째 원소 = meta7 텐서 → stack
stack_meta_collate = make_collate(meta_index=2, meta_mode="stack", meta_optional=False)
# RespireNet: 3번째 원소 = 청크 인덱스 int → long
index_meta_collate = make_collate(meta_index=2, meta_mode="as_tensor", meta_dtype=torch.long, meta_optional=False)
# PC-MCL 학습: y 가 이미 텐서(3-label float | 4-class long), 3번째 원소 = ssl_label 텐서
pcmcl_train_collate = make_collate(label_mode="stack", meta_index=2, meta_mode="stack", meta_optional=False)

__all__ = [
    "collate_xy_meta", "make_collate", "tuple_collate", "qlung_collate", "btscard_sprs_collate",
    "stack_meta_collate", "index_meta_collate", "pcmcl_train_collate",
]
