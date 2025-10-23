from abc import ABC, abstractmethod
import torch
import pathlib
import torchaudio
import pandas as pd


from rsc_toolkit.datasets.cycle import Cycle


class BaseCycleStore(ABC):
    """Abstract interface for storing and retrieving cycles."""

    def __init__(self, filename: str = "cycle_index.csv"):
        self._index_records = []
        self.filename = filename
        self.load_index()

    @property # TODO cached
    def index(self):
        return pd.DataFrame(self._index_records)

    @abstractmethod
    def save_cycle(self, cycle_id: str, cycle: Cycle) -> None:
        pass

    @abstractmethod
    def load_cycle(self, cycle_id: str) -> Cycle:
        pass

    def add_index_entry(self, record: dict):
        self._index_records.append(record)

    def save_index(self) -> pd.DataFrame:
        """Finalize and optionally persist the index to disk."""
        path = self.root / self.filename
        self.index.to_csv(path, index=False)

    def load_index(self) -> pd.DataFrame:
        """Load an existing index CSV from disk."""
        path = self.root / self.filename
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        df = pd.read_csv(path)
        self._index_records = df.to_dict(orient="records")


class FileCycleStore(BaseCycleStore):
    """Save the cycles as individual unique files."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        super().__init__()

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
