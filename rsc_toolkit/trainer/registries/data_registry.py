"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.DATAMODULES.

Built-in entries are declared lazily in the hub and imported only on first lookup, so a missing
optional dependency (librosa, etc.) of one datamodule does not break importing this file. Register a new
datamodule in the hub declaration or with the `@DATAMODULES.register("name")` decorator.
"""
from rsc_toolkit.registries import DATAMODULES

__all__ = ["DATAMODULES"]
