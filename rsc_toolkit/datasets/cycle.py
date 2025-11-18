from typing import Dict, Any
from abc import ABC, abstractmethod
import pathlib

from pydantic import ConfigDict, BaseModel
import torch
import torchaudio


class Cycle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: torch.Tensor                # sliced waveform
    start: float                       # seconds
    end: float                         # seconds

    @classmethod
    def from_dict(cls, data: dict):
        """Prepare for deserialization from disk."""
        return cls(**data)
