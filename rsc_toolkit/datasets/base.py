from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging


logger = logging.getLogger(__name__)


class BaseDatasetDownloader(ABC, BaseModel):
    """
    Interface for downloading datasets
    """
    name: str

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
        if self.validate_dataset_files():
            logger.info(f"✅ Found existing dataset {self.name}, skipping download.")
            return
        logger.info(f"⬇️ Downloading dataset {self.name}")
        self.download_dataset()
        

    
class BaseDataset(ABC, BaseModel):
    """
    Interface datasets
    """

    downloader: type[BaseDatasetDownloader]

    def get_downloader(self, **kwargs):
        return self.downloader(**kwargs)