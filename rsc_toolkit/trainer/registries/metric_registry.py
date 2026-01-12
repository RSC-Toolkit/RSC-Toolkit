from typing import Dict, Type
from torchmetrics import Metric

from rsc_toolkit.metrics.topk_accuracy import TopKAccuracy
from rsc_toolkit.metrics.icbhi.score import ICBHIScore


METRICS: Dict[str, Type[Metric]] = {
    "topk_accuracy": TopKAccuracy,
    "icbhi_score": ICBHIScore,
}