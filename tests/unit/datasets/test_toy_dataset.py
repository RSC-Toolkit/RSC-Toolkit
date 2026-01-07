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


def test_toy_dataset_build_index(dataset):
    dataset.download()
    index = dataset.processor.build_index(sample_level="cycle")
    assert len(index.records) > 0
    index_path = dataset.processor.index_dir_path / "index_cycle.csv"
    assert index_path.exists()
    stats_path = dataset.processor.index_dir_path / "index_cycle_stats.json"
    assert stats_path.exists()


def test_toy_dataset_index_csv_headers(dataset):
    dataset.download()
    dataset.processor.build_index(sample_level="clip")
    index_path = dataset.processor.index_dir_path / "index_clip.csv"
    assert index_path.exists()
    header = index_path.read_text().splitlines()[0]
    assert header.split(",") == [
        "sample_id",
        "recording_path",
        "sample_level",
        "t0",
        "t1",
        "label",
        "meta",
    ]


def test_toy_dataset_validate_raw_files(dataset):
    dataset.download()
    assert dataset.downloader.validate_downloaded_files()
