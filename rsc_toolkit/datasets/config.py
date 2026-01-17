from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    data_dir: Optional[str] = None
    force: bool = False


class ProcessorConfig(BaseModel):
    """
    Minimal processor config placeholder.

    The intent is that this config fully determines the processor signature, so
    it can be safely used for caching and experiment reproduction.
    """

    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    mode: Literal["off", "read", "readwrite"] = "readwrite"
    strict: bool = False
    cache_dir: Optional[str] = None


class LoaderConfig(BaseModel):
    sample_level: Literal["cycle", "clip"] = "cycle"
    cache: CacheConfig = Field(default_factory=CacheConfig)
