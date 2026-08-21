from __future__ import annotations

from typing import ClassVar, Type

from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.datasets import DatasetFactory

from .downloader import HFLungV1DatasetDownloader
from .processor import HFLungV1DatasetProcessor


@DatasetFactory.register_dataset_class("hflungv1")
class HFLungV1Dataset(BaseRSCDataset):
    downloader_class: ClassVar[Type[HFLungV1DatasetDownloader]] = HFLungV1DatasetDownloader
    processor_class: ClassVar[Type[HFLungV1DatasetProcessor]] = HFLungV1DatasetProcessor
