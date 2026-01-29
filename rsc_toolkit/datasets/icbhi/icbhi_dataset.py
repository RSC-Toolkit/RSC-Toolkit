from __future__ import annotations

from typing import ClassVar, Type, Literal, Optional
import pathlib

from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.datasets import DatasetFactory

from .downloader import ICBHIDatasetDownloader
from .processor import ICBHIDatasetProcessor


@DatasetFactory.register_dataset_class("icbhi")
class ICBHIDataset(BaseRSCDataset):
    downloader_class: ClassVar[Type[ICBHIDatasetDownloader]] = ICBHIDatasetDownloader
    processor_class: ClassVar[Type[ICBHIDatasetProcessor]] = ICBHIDatasetProcessor
