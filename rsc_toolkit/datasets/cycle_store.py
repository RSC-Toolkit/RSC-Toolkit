from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type, TypeVar
import hashlib
import uuid
import torch
from rsc_toolkit.datasets.cycle import Cycle

T = TypeVar("T", bound=Cycle)


def _create_deterministic_key_for_tensor(t: torch.Tensor) -> str:
    data_bytes = t.detach().cpu().numpy().tobytes()
    h = hashlib.sha256(data_bytes).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, h))


def _create_random_key() -> str:
    return str(uuid.uuid4())


def _create_key(tensor: torch.Tensor, allow_duplicates: bool) -> str:
    if allow_duplicates:
        return _create_random_key()
    return _create_deterministic_key_for_tensor(tensor)


class BaseCycleStore(ABC):
    """Abstract interface for storing and retrieving cycles."""

    def __init__(self, path: Path):
        self.path = path
        self._index: set[str] = set()

    @property
    def index(self) -> set[str]:
        return self._index

    def _add_index_entry(self, key: str) -> None:
        self._index.add(key)

    @abstractmethod
    def add_cycle(self, cycle: Cycle, key: Optional[str] = None) -> str:
        ...

    @abstractmethod
    def get_cycle(self, key: str, cls: Type[T] = Cycle) -> T:
        ...


class FileCycleStore(BaseCycleStore):
    """Save cycles as individual .pt files in a directory."""

    def __init__(self, path: Path, allow_duplicates: bool = False):
        super().__init__(path)
        self.allow_duplicates = allow_duplicates
        path.mkdir(parents=True, exist_ok=True)
        self._populate_index()

    def _populate_index(self) -> None:
        for p in self.path.glob("*.pt"):
            self._add_index_entry(p.stem)

    def _key_to_path(self, key: str) -> Path:
        return self.path / f"{key}.pt"

    def add_cycle(self, cycle: Cycle, key: Optional[str] = None) -> str:
        if key is None:
            key = _create_key(cycle.audio, self.allow_duplicates)

        if not self.allow_duplicates and key in self._index:
            raise ValueError(f"Duplicate cycle detected (key={key})")

        path = self._key_to_path(key)
        torch.save(cycle.model_dump(), path)
        self._add_index_entry(key)
        return key

    def get_cycle(self, key: str, cls: Type[T] = Cycle) -> T:
        path = self._key_to_path(key)
        if not path.exists():
            raise KeyError(f"No cycle found for key: {key}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**data)
