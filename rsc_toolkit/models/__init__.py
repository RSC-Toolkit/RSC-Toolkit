# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Model implementations.

Each module registers its models with `@ModelFactory.register_model_class("<name>")`;
importing this package imports every module, which completes registration.
Look models up via ModelFactory / create_model / build_model(trainer).
"""
from rsc_toolkit.models.factory import ModelFactory, create_model

ModelFactory.register_models()

__all__ = ["ModelFactory", "create_model"]
