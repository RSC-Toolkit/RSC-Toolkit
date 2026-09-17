"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.MODELS (decorators in each model module).

This only keeps the `MODELS` name used by build_model; no models are listed here.
To add a model, put a module under rsc_toolkit/models/ and register it with the decorator.
"""
from rsc_toolkit.registries import MODELS

__all__ = ["MODELS"]
