# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Registry of raw datasets (download, index, CLI) -- backward-compatible entry point.

The source of truth is `rsc_toolkit.registries.DATASETS` (shared Registry). Dataset modules register with a decorator:

    from rsc_toolkit.datasets.factory import DatasetFactory
    @DatasetFactory.register_dataset_class("icbhi")
    class ICBHIDataset(BaseRSCDataset): ...

    from rsc_toolkit.registries import DATASETS   # same thing
    @DATASETS.register("heartsound")
    class HeartSoundDataset(BaseRSCDataset): ...

The 4 built-in datasets (icbhi, sprsound, hflungv1, toy) are declared lazily in the hub, so they are always visible regardless of file naming.
Scanning package modules named `*_dataset*` (register_datasets) is kept for backward compatibility with the old convention; external packages
attach via the `rsc_toolkit.plugins` entry point / `RSC_TOOLKIT_PLUGINS` (see rsc_toolkit.registry).
Registering a different class under an existing name raises ValueError (no silent overwrite).
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Callable, List, Type

from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.registries import DATASETS
from rsc_toolkit.registry import Registry, load_plugins

logger = logging.getLogger(__name__)

_SCANNED = False


class DatasetFactory:
    # Mapping (keys() are the CLI choices) -- entry-point datasets show up automatically.
    datasets: Registry = DATASETS

    @staticmethod
    def register_datasets() -> None:
        """
        Import modules under rsc_toolkit.datasets whose names contain "_dataset" (subpackages included) so their
        decorators run, then load plugins. Idempotent. Import failures only log a warning per module.
        """
        global _SCANNED
        if _SCANNED:
            return
        _SCANNED = True

        import rsc_toolkit.datasets as package

        def _try_import(name: str) -> None:
            try:
                importlib.import_module(name)
            except Exception as exc:
                # Swallowing only ImportError would let an internal typo (NameError, etc.) abort the whole scan -- isolate per module + record
                # (same principle as registry.Registry._scan_package).
                DATASETS._failed[name] = f"{type(exc).__name__}: {exc}"
                logger.log(logging.WARNING if isinstance(exc, ImportError) else logging.ERROR,
                           "Dataset module %s could not be imported and is excluded from registration (%s): %s", name,
                           "missing optional dependency?" if isinstance(exc, ImportError) else "error inside the module", exc)

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if "_dataset" in module_name:
                _try_import(f"rsc_toolkit.datasets.{module_name}")
                continue
            if not is_pkg:
                continue
            try:
                subpackage = importlib.import_module(f"rsc_toolkit.datasets.{module_name}")
            except ImportError as exc:
                logger.warning("Could not import dataset subpackage %s: %s", module_name, exc)
                continue
            if not hasattr(subpackage, "__path__"):
                continue
            for _, submodule_name, _ in pkgutil.iter_modules(subpackage.__path__):
                if "_dataset" in submodule_name:
                    _try_import(f"rsc_toolkit.datasets.{module_name}.{submodule_name}")

        load_plugins()

    @classmethod
    def register_dataset_class(cls, dataset_name: str) -> Callable[[Type[BaseRSCDataset]], Type[BaseRSCDataset]]:
        """Class decorator. Re-registering a different class under the same name raises ValueError (same class is idempotent)."""
        return DATASETS.register(dataset_name)

    @classmethod
    def get_dataset_names(cls) -> List[str]:
        return list(DATASETS)

    @classmethod
    def get_dataset_class(cls, dataset_name: str) -> Type[BaseRSCDataset]:
        return DATASETS.get(dataset_name)  # unknown name: KeyError("Unknown dataset '...'. Available: [...]")
