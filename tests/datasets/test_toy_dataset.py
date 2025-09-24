from pydantic import PrivateAttr
import pytest
import pathlib


from rsc_toolkit.datasets.toy_dataset import ToyDataset

@pytest.fixture
def dataset(tmp_path):
    outputs_directory = str(tmp_path)
    force = True
    downloader = ToyDataset.make_downloader(outputs_directory=outputs_directory, force=force)
    processor = ToyDataset.make_processor(outputs_directory=outputs_directory, force=force)
    dataset = ToyDataset(
        downloader=downloader,
        processor=processor,
    )
    return dataset

def test_toy_dataset(dataset):
    dataset.download()
    dataset.process()
