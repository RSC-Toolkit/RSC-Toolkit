import pathlib
import logging
import functools


# from rsc_toolkit.datasets.icbhi_dataset import ICBHIDataset, ICBHIDatasetProcessor
# from rsc_toolkit.datasets import DatasetFactory


# logger = logging.getLogger(__name__)


# ToyDatasetProcessor = ICBHIDatasetProcessor

# @DatasetFactory.register_dataset_class("toy")
# class ToyDataset(ICBHIDataset):
#     """
#     Toy Dataset for testing.

#     Generated from ICHBI, and relying on ICBHI files (The subset is provided and does not need download).
#     """

#     def download(self, **kwargs):
#         logger.info("Toy dataset already provided, no download needed.")
