"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.OPTIMIZERS (built-ins declared lazily, external entries via decorator)."""
from rsc_toolkit.registries import OPTIMIZERS

__all__ = ["OPTIMIZERS"]
