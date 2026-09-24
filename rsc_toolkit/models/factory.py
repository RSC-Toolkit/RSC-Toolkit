# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Model registry — backward-compatibility facade.

The source of truth for registration is `rsc_toolkit.registries.MODELS` (the shared Registry). This module keeps the public API
(`ModelFactory.register_model_class` / `registry` / `get_model_names` / `get_model_class` /
`create_model` / `register_models`, constants `PLUGIN_ENTRY_POINT_GROUP` / `PLUGIN_ENV_VAR`) with the same signatures and
exception semantics. New code reaches the same registry through either:

    from rsc_toolkit.models.factory import ModelFactory
    @ModelFactory.register_model_class("my_model")
    class MyModel(BaseRSCModel, nn.Module): ...

    from rsc_toolkit.registries import MODELS          # equivalent
    @MODELS.register("my_model")
    class MyModel(BaseRSCModel, nn.Module): ...

`import rsc_toolkit.models` imports every module under this package (register_models), so
placing a module inside the package is enough to register it. `model.name` in the config files (configs/model/*.yaml)
must match a name registered here, and build_model looks up the class in this registry.

There are three registration paths, processed in this order:

1. **In-package scan** — modules inside `rsc_toolkit/models/` are imported automatically.
2. **Entry points** (for external packages). Adding this to your package's `pyproject.toml`
   registers the models on install — no need to fork the toolkit:

       [project.entry-points."rsc_toolkit.plugins"]     # group shared by all component kinds
       my_models = "my_pkg.models"

   The legacy group `rsc_toolkit.models` is still read. The value must point to a module (or callable)
   whose import runs the decorators.
3. **`RSC_TOOLKIT_PLUGINS` environment variable** — an escape hatch for local modules that are not packaged yet.
   Imports a comma-separated list of module paths:

       RSC_TOOLKIT_PLUGINS=my_models,lab.experiments.arch rsc-toolkit-train ...

The contract a model must follow is `BaseRSCModel` in `rsc_toolkit/models/base.py`.
"""
from __future__ import annotations

from typing import Callable, List

from rsc_toolkit.registries import MODELS
from rsc_toolkit.registry import PLUGIN_ENV_VAR, Registry, load_plugins

# Legacy entry-point group name used by external packages to register models (backward compatibility; the shared group is
# rsc_toolkit.registry.PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.plugins").
PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.models"

__all__ = ["ModelFactory", "create_model", "PLUGIN_ENTRY_POINT_GROUP", "PLUGIN_ENV_VAR"]


class ModelFactory:
    @staticmethod
    def register_models() -> None:
        """
        Imports every module under rsc_toolkit.models so the decorators run, and loads plugins. Idempotent.

        Modules that fail to import because an optional dependency is missing are skipped with a warning —
        one model failing to load must not make the others unusable.
        """
        MODELS._discover()

    @staticmethod
    def _register_plugins() -> None:
        """(Re)loads entry-point and environment-variable plugins. One failing plugin does not stop the others."""
        load_plugins(force=True)

    @classmethod
    def register_model_class(cls, model_name: str) -> Callable[[type], type]:
        """Class decorator. Registering a different class under an existing name fails."""
        return MODELS.register(model_name)

    @classmethod
    def registry(cls) -> Registry:
        """Name -> class registry (a live object; later registrations are reflected). Supports Mapping + pop."""
        MODELS._discover()
        return MODELS

    @classmethod
    def get_model_names(cls) -> List[str]:
        return sorted(cls.registry())

    @classmethod
    def get_model_class(cls, model_name: str) -> type:
        return cls.registry().get(model_name)  # unknown name: KeyError("Unknown model '...'. Available: [...]")

    @classmethod
    def create_model(cls, model_name: str, **params):
        return cls.get_model_class(model_name)(**params)


def create_model(model_name: str, **params):
    """Convenience function. Same as ModelFactory.create_model."""
    return ModelFactory.create_model(model_name, **params)
