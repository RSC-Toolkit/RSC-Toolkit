"""Adapter (LightningModule wrapper) selection.

Names are looked up in rsc_toolkit.registries.ADAPTERS. A typo ('split-iter', etc.) raises KeyError with the
list of available names instead of silently falling back to PLAdapter, so that results are never recorded
without M2D's split-batch quirk.
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
    adapter_cls = ADAPTERS.get(name)  # unknown name: KeyError("Unknown adapter '...'. Available: [...]")
    params = dict(adapter_cfg.get("params") or {})
    return adapter_cls(
        rsc_model=rsc_model,
        loss_func=loss_func,
        train_metric_fns=train_metric_fns,
        val_metric_fns=val_metric_fns,
        optimizer_cfg=optimizer_cfg,
        **params,
    )
