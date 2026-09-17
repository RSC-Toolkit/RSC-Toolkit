from functools import partial
from typing import Any, Callable, Dict, List

from torchmetrics import Metric

from rsc_toolkit.registries import METRICS


MetricFactory = Callable[[], Metric]


def build_metric_factories(cfg_list: List[Dict[str, Any]]) -> List[MetricFactory]:
    """
    cfg_list example:
      [
        {"name": "topk_accuracy", "params": {"topk": (1,)}},
        {"name": "icbhi_score", "params": {"num_classes": 4, "normal_index": 0, "topk": 1}},
      ]

    Metrics must be owned by the LightningModule, so only factories (partial) are created here, not instances.
    An unknown name raises KeyError listing the available names.
    """
    factories: List[MetricFactory] = []
    for item in cfg_list:
        if "name" not in item or item["name"] is None:
            raise KeyError(f"metric config has no 'name'. Got keys: {sorted(item)}")
        cls = METRICS.get(item["name"])
        params = dict(item.get("params") or {})
        factories.append(partial(cls, **params))  # late construct, owned by LightningModule
    return factories


def build_train_val_metric_factories(cfg: Dict[str, Any]) -> tuple[List[MetricFactory], List[MetricFactory]]:
    """
    cfg example:
      {"train": [...], "val": [...]}
    """
    # Reject typo keys such as 'vali:'/'valid:'/'test:'; silently ignoring them would drop all metrics
    # for that split, and the best-checkpoint callback would produce nothing.
    unknown = set(cfg) - {"train", "val"}
    if unknown:
        raise KeyError(
            f"unknown keys in metrics config {sorted(unknown)}; the only allowed top-level keys are 'train' and 'val'"
        )
    train_cfg = cfg.get("train", []) or []
    val_cfg = cfg.get("val", []) or []
    return build_metric_factories(train_cfg), build_metric_factories(val_cfg)
