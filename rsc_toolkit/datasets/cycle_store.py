from typing import Dict, Any
from abc import ABC, abstractmethod
from pydantic import ConfigDict, BaseModel
import torch
import pathlib
import torchaudio
import pandas as pd


from rsc_toolkit.datasets.cycle import Cycle


class BaseCycleStore(ABC):
    """Abstract interface for storing and retrieving cycles."""

    def __init__(self):
        self._index_records = []

    @abstractmethod
    def save_cycle(self, cycle_id: str, cycle: Cycle) -> None:
        pass

    @abstractmethod
    def load_cycle(self, cycle_id: str) -> Cycle:
        pass

    def add_index_entry(self, record: dict):
        self._index_records.append(record)

    def finalize_index(self) -> pd.DataFrame:
        return pd.DataFrame(self._index_records)


class FileCycleStore(BaseCycleStore):
    """Save the cycles as individual unique files."""

    def __init__(self, root: pathlib.Path):
        super().__init__()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_cycle(self, cycle_id: str, cycle: Cycle):
        path = self.root / f"{cycle_id}.pt"
        torch.save(cycle.to_dict(), path)
        entry = cycle.to_index_entry()
        entry['cycle_id'] = cycle_id
        entry['path'] = path
        self.add_index_entry(entry)

    def load_cycle(self, cycle_id: str, cls: type = Cycle) -> Cycle:
        data = torch.load(self.root / f"{cycle_id}.pt", map_location="cpu")
        return cls.from_dict(data)

    def index(self):
        return [rec["cycle_id"] for rec in self._index_records]
