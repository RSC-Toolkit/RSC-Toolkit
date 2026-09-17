"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.LOSSES (built-ins declared lazily, external entries via decorator)."""
from rsc_toolkit.registries import LOSSES

__all__ = ["LOSSES"]
