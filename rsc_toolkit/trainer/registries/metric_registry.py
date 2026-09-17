"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.METRICS (built-ins declared lazily, external entries via decorator)."""
from rsc_toolkit.registries import METRICS

__all__ = ["METRICS"]
