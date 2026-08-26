from typing import Dict, Type
from torchmetrics import Metric

from rsc_toolkit.metrics.topk_accuracy import TopKAccuracy
from rsc_toolkit.metrics.icbhi.score import ICBHIScore
from rsc_toolkit.metrics.icbhi.binary_score import PaperBinaryScore


METRICS: Dict[str, Type[Metric]] = {
    "topk_accuracy": TopKAccuracy,
    "icbhi_score": ICBHIScore,
    "icbhi_paper_binary": PaperBinaryScore,  # AMS-D 논문 이진(비표준) Se/Sp/Score (CcnNnn/AMS-D utils/metric.py)
}