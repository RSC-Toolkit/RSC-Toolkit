# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.DATAMODULES.

Built-in entries are declared lazily in the hub and imported only on first lookup, so a missing
optional dependency (librosa, etc.) of one datamodule does not break importing this file. Register a new
datamodule in the hub declaration or with the `@DATAMODULES.register("name")` decorator.
"""
from rsc_toolkit.registries import DATAMODULES

__all__ = ["DATAMODULES"]
