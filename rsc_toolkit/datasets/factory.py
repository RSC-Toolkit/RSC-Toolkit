from typing import ClassVar, List, Dict, Type
import importlib
import pkgutil
import sys


from rsc_toolkit.datasets.base import BaseRSCDataset


class DatasetFactory:
    datasets: ClassVar[Dict[str, Type[BaseRSCDataset]]] = {}

    @staticmethod
    def register_datasets():
        """
        We register datasets by importing modules that are under this module with "_dataset" in path.
        Inside modules, must apply decorator:
        from rsc_toolkit.datasets import DatasetFactory
        @DatasetFactory.register_dataset_class("icbhi")
        """
        package = sys.modules[__package__]  # this gives rsc_toolkit.datasets
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if "_dataset" in module_name:
                importlib.import_module(f"rsc_toolkit.datasets.{module_name}")
                continue

            if not is_pkg:
                continue

            subpackage = importlib.import_module(f"rsc_toolkit.datasets.{module_name}")
            if not hasattr(subpackage, "__path__"):
                continue

            for _, submodule_name, _ in pkgutil.iter_modules(subpackage.__path__):
                if "_dataset" in submodule_name:
                    importlib.import_module(
                        f"rsc_toolkit.datasets.{module_name}.{submodule_name}"
                    )

    @classmethod
    def register_dataset_class(cls, dataset_name: str):
        """Decorator to register a dataset class."""
        def decorator(dataset_cls: Type[BaseRSCDataset]):
            cls.datasets[dataset_name] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def get_dataset_names(cls) -> List[str]:
        return list(cls.datasets.keys())

    @classmethod
    def get_dataset_class(cls, dataset_name: str) -> Type[BaseRSCDataset]:
        return cls.datasets[dataset_name]
