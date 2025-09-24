from typing import ClassVar
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging
import functools
import pathlib
import hashlib
import json


logger = logging.getLogger(__name__)


class BaseRSCDatasetDownloader(ABC, BaseModel):
    """
    Interface for downloading datasets
    """
    name: str
    outputs_directory: str
    force: bool = False

    @abstractmethod
    def validate_dataset_files(self) -> bool:
        """
        Validate the dataset files.
        """

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Download the dataset files
        """

    def download(self):
        if not self.force and self.validate_dataset_files():
            logger.info(f"Found existing {self.name}, skipping download.")
            return
        logger.info(f"⬇️ Downloading {self.name}")
        self.download_dataset()
        logger.info(f"✅ Completed {self.name} download")
        


class BaseRSCDatasetProcessor(ABC, BaseModel):
    """
    Interface for processing datasets for training
    """
    name: str # TODO better to use this from parent?
    outputs_directory: str
    force: bool = False

    @functools.cached_property
    def cache_dir_path(self):

        def _create_unique_name(**kwargs):
            # Sort keys for determinism
            serialized = json.dumps(kwargs, sort_keys=True, default=str)
            # Hash for compact unique identifier
            return hashlib.md5(serialized.encode()).hexdigest()
        
        # put settings that should distinguish in here
        key = _create_unique_name(

        )

        # create cache/dataset/DatasetName dir for temp files.
        cache_path = pathlib.Path(self.outputs_directory, "dataset", self.name, key)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @abstractmethod
    def validate_dataset_files(self) -> bool:
        """
        Validate preocessed files files
        """

    @abstractmethod
    def process_dataset(self) -> None:
        """
        Process the dataset for running
        """

    def process(self):
        if not self.force and self.validate_dataset_files():
            logger.info(f"Found existing {self.name}, skipping download.")
            return
        logger.info(f"⬇️ Processing {self.name}")
        self.process_dataset()
        logger.info(f"✅ Completed {self.name} processing")

    
class BaseRSCDataset(ABC, BaseModel):
    """
    Interface datasets
    """

    downloader_class: ClassVar[type[BaseRSCDatasetDownloader]]
    processor_class: ClassVar[type[BaseRSCDatasetProcessor]]

    downloader: BaseRSCDatasetDownloader
    processor: BaseRSCDatasetProcessor

    @classmethod
    def make_downloader(cls, **kwargs) -> BaseRSCDatasetDownloader:
        return cls.downloader_class(**kwargs)

    @classmethod
    def make_processor(cls, **kwargs) -> BaseRSCDatasetProcessor:
        return cls.processor_class(**kwargs)

    def download(self):
        self.downloader.download()

    def process(self):
        self.processor.process()


