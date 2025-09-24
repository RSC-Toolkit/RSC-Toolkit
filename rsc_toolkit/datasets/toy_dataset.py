import pathlib
import logging
import functools


from rsc_toolkit.datasets.base import BaseDatasetDownloader, BaseDataset
from rsc_toolkit.datasets import DatasetFactory
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


@DatasetFactory.register_dataset_class("toy")
class ToyDataset(BaseDataset):
    """
    Toy Dataset for testing.

    Generated from ICHBI, and relying on ICBHI files (The subset is provided and does not need download).
    """

    def download(self, **kwargs):
        logger.info("Toy dataset already provided, no download needed.")
