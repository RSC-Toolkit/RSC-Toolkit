from functools import partial
from typing import Any, Callable, Dict, List

from torchmetrics import Metric

from rsc_toolkit.trainer.registries.metric_registry import METRICS


MetricFactory = Callable[[], Metric]


def build_metric_factories(cfg_list: List[Dict[str, Any]]) -> List[MetricFactory]:
    """
    cfg_list example:
      [
        {"name": "topk_accuracy", "params": {"topk": (1,)}},
        {"name": "icbhi_score", "params": {"num_classes": 4, "normal_index": 0, "topk": 1}},
      ]
    """
    factories: List[MetricFactory] = []
    for item in cfg_list:
        name = item["name"]
        params = item.get("params", {})

        if name not in METRICS:
            raise KeyError(f"Unknown metric '{name}'. Available: {sorted(METRICS.keys())}")

        cls = METRICS[name]
        factories.append(partial(cls, **params))  # late construct, owned by LightningModule

    return factories


def build_train_val_metric_factories(cfg: Dict[str, Any]) -> tuple[List[MetricFactory], List[MetricFactory]]:
    """
    cfg example:
      {"train": [...], "val": [...]}
    """
    # 'vali:'/'valid:'/'test:' 같은 오타 키는 예전엔 침묵으로 무시됐다 — 그 스플릿의
    # 메트릭이 통째로 빠진 채 학습이 완주하고, best-checkpoint 콜백의 무산출로 이어졌다.
    unknown = set(cfg) - {"train", "val"}
    if unknown:
        raise KeyError(
            f"metrics config 의 알 수 없는 키 {sorted(unknown)} — 허용되는 최상위 키는 'train', 'val' 뿐이다"
        )
    train_cfg = cfg.get("train", []) or []
    val_cfg = cfg.get("val", []) or []
    return build_metric_factories(train_cfg), build_metric_factories(val_cfg)
