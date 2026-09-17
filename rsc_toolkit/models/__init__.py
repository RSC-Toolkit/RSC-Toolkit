"""
Model implementations.

Each module registers its models with `@ModelFactory.register_model_class("<name>")`;
importing this package imports every module, which completes registration.
Look models up via ModelFactory / create_model / build_model(trainer).
"""
from rsc_toolkit.models.factory import ModelFactory, create_model

ModelFactory.register_models()

__all__ = ["ModelFactory", "create_model"]
