from typing import ClassVar, List
import importlib
import pkgutil
import sys


from rsc_toolkit.models.base import BaseRSCModel


MODELS = {}

class ModelFactory:
    

    @staticmethod
    def register_models():
        """
        We register models by importing modules that are under this module with "_model" in path.
        Inside modules, must apply decorator:
        from rsc_toolkit.models import ModelFactory
        @ModelFactory.register_model_class("bts")
        """
        package = sys.modules[__package__]  # this gives rsc_toolkit.models
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if "_model" in module_name:
                importlib.import_module(f"rsc_toolkit.models.{module_name}")

    @classmethod
    def register_model_class(cls, model_name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: BaseRSCModel):
            MODELS[model_name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_model_names(cls) -> List[str]:
        return list(MODELS.keys())

    @classmethod
    def get_model_class(cls, model_name: str) -> BaseRSCModel:
        return MODELS[model_name]
