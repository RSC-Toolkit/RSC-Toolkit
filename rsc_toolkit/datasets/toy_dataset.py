import logging
import pathlib
from typing import ClassVar, Optional, Type


from rsc_toolkit.datasets.icbhi import ICBHIDataset
from rsc_toolkit.datasets.icbhi.icbhi_dataset import ICBHIDatasetProcessor
from rsc_toolkit.datasets import DatasetFactory
from rsc_toolkit.datasets.base import BaseRSCDatasetDownloader


logger = logging.getLogger(__name__)


class ToyDatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "toy"

    def download_dataset(self) -> None:
        logger.info("Toy dataset is bundled with the repository -> no download.")

    def validate_downloaded_files(self) -> bool:
        return True # Toy dataset is bundled with the repository, no validation needed.


class ToyDatasetProcessor(ICBHIDatasetProcessor):
    name: str = "toy"
    raw_data_directory: Optional[str] = None

    def validate_processed_files(self) -> bool:
        return True # Toy dataset is bundled with the repository, no validation needed.

    @property
    def data_folder(self):
        if self.raw_data_directory:
            return pathlib.Path(self.raw_data_directory)

        project_root = pathlib.Path(__file__).resolve().parents[2]
        sample_data_path = project_root / "sample_data" / "toy_dataset" / "files"
        if not sample_data_path.exists():
            raise ValueError(f"Toy dataset not found at {sample_data_path}. Set raw_data_directory explicitly.")
        return sample_data_path


@DatasetFactory.register_dataset_class("toy")
class ToyDataset(ICBHIDataset):
    """
    Toy Dataset for testing.

    Generated from ICHBI, and relying on ICBHI files (The subset is provided and does not need download).
    """

    downloader_class: ClassVar[Type[ToyDatasetDownloader]] = ToyDatasetDownloader
    processor_class: ClassVar[Type[ToyDatasetProcessor]] = ToyDatasetProcessor
