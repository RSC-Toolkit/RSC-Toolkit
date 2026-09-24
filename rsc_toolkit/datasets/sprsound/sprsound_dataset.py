from __future__ import annotations

from typing import ClassVar, Type

from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.datasets import DatasetFactory

from .downloader import SPRSoundDatasetDownloader
from .processor import SPRSoundDatasetProcessor


@DatasetFactory.register_dataset_class("sprsound")
class SPRSoundDataset(BaseRSCDataset):
    downloader_class: ClassVar[Type[SPRSoundDatasetDownloader]] = SPRSoundDatasetDownloader
    processor_class: ClassVar[Type[SPRSoundDatasetProcessor]] = SPRSoundDatasetProcessor
