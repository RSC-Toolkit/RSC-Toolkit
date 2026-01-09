from __future__ import annotations

from typing import ClassVar, Type, Literal, Optional
import pathlib

from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.datasets import DatasetFactory

from .downloader import ICBHIDatasetDownloader
from .processor import ICBHIDatasetProcessor
from .index_dataset import ICBHIIndexDataset


@DatasetFactory.register_dataset_class("icbhi")
class ICBHIDataset(BaseRSCDataset):
    downloader_class: ClassVar[Type[ICBHIDatasetDownloader]] = ICBHIDatasetDownloader
    processor_class: ClassVar[Type[ICBHIDatasetProcessor]] = ICBHIDatasetProcessor

    # Optional convenience constructor
    @staticmethod
    def torch_dataset(
        index_csv: str | pathlib.Path,
        data_root: Optional[str | pathlib.Path] = None,
        sampling_rate: int = 10000,
        sample_level: Optional[Literal["cycle", "clip"]] = None,
        split: Optional[Literal["train", "test"]] = None,
        split_file: Optional[str | pathlib.Path] = None,
    ) -> ICBHIIndexDataset:
        return ICBHIIndexDataset(
            index_csv=index_csv,
            data_root=data_root,
            sampling_rate=sampling_rate,
            sample_level=sample_level,
            split=split,
            split_file=split_file,
        )
