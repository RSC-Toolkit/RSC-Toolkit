from typing import Dict, Any
from abc import ABC, abstractmethod
from pydantic import ConfigDict, BaseModel
import torch
import pathlib
import torchaudio
import pandas as pd


class Cycle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: torch.Tensor                # sliced waveform
    start: float                       # seconds
    end: float                         # seconds

    @property
    def label(self):
        """Compute label from annotation if not already assigned."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_index_entry(self) -> dict:
        """Prepare data to show in index."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict):
        """Prepare for deserialization from disk."""
        return cls(**data)
