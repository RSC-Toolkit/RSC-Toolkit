from typing import Dict, Type

import torch.nn as nn

from rsc_toolkit.models.clap import AudioCLAP


MODELS: Dict[str, Type[nn.Module]] = {
    "audioclap": AudioCLAP,
}
