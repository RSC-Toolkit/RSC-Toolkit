"""adapter(LightningModule 래퍼) 선택 — 예전 train.py 의 `if name == "split_iter"` 문을 대체한다.

이름은 rsc_toolkit.registries.ADAPTERS 에서 찾는다. 오타('split-iter' 등)는 예전처럼 PLAdapter 로
침묵 폴백하지 않고 가용 목록과 함께 KeyError 를 낸다 — M2D 의 분할 배치 quirk 가 빠진 채
재현 수치가 기록되는 사고를 막기 위해서다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightning as L
import torch.nn as nn

from rsc_toolkit.registries import ADAPTERS

DEFAULT_ADAPTER = "pl"


def build_adapter(
    adapter_cfg: Optional[Dict[str, Any]],
    *,
    rsc_model: nn.Module,
    loss_func: nn.Module,
    optimizer_cfg: Dict[str, Any],
    train_metric_fns: Optional[List[Any]] = None,
    val_metric_fns: Optional[List[Any]] = None,
) -> L.LightningModule:
    """
    adapter_cfg example:
      None / {}                                        -> PLAdapter
      {"name": "split_iter", "params": {"split_iter": 4}} -> SplitIterAdapter(split_iter=4)
    """
    adapter_cfg = dict(adapter_cfg or {})
    name = adapter_cfg.get("name") or DEFAULT_ADAPTER
    adapter_cls = ADAPTERS.get(name)  # 미지 이름: KeyError("Unknown adapter '...'. Available: [...]")
    params = dict(adapter_cfg.get("params") or {})
    return adapter_cls(
        rsc_model=rsc_model,
        loss_func=loss_func,
        train_metric_fns=train_metric_fns,
        val_metric_fns=val_metric_fns,
        optimizer_cfg=optimizer_cfg,
        **params,
    )
