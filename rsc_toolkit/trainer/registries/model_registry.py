from typing import Dict, Type
import logging

import torch.nn as nn

from rsc_toolkit.models.clap import AudioCLAP
from rsc_toolkit.models.cnn import CNNClassifier
from rsc_toolkit.models.resnet import ResNetClassifier
from rsc_toolkit.models.ast import ASTClassifier

logger = logging.getLogger(__name__)


MODELS: Dict[str, Type[nn.Module]] = {
    "audioclap": AudioCLAP,
    "ast": ASTClassifier
    "cnn": CNNClassifier,
    "resnet": ResNetClassifier,
}