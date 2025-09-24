from typing import Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch


from rsc_toolkit.augmentations import Augmentation


class BaseAudioProcessor(BaseModel, ABC):
    augmentations: Optional[Augmentation] = None

    @abstractmethod
    def make_features(self, audio: torch.Tensor):
        """Convert one audio sample into features"""
        pass
