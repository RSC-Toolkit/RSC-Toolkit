from typing import Dict, Type
import logging

import torch.nn as nn

from rsc_toolkit.models.clap import AudioCLAP
from rsc_toolkit.models.cnn import CNNClassifier

logger = logging.getLogger(__name__)


MODELS: Dict[str, Type[nn.Module]] = {
    "audioclap": AudioCLAP,
    "cnn": CNNClassifier,
}

try:
    from rsc_toolkit.models.ast import ASTClassifier

    MODELS["ast"] = ASTClassifier
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("AST model not registered: %s", exc)
