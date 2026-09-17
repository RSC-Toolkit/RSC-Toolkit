# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""Backward-compatibility alias. The source of truth is rsc_toolkit.registries.LOSSES (built-ins declared lazily, external entries via decorator)."""
from rsc_toolkit.registries import LOSSES

__all__ = ["LOSSES"]
