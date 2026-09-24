# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
This test module checks the common behaviours of datasets
"""

from pydantic import PrivateAttr


from rsc_toolkit.datasets.base import (
    BaseRSCDatasetDownloader,
    BaseRSCDatasetProcessor,
    BaseRSCDataset,
)


class DummyDownloader(BaseRSCDatasetDownloader):
    _valid: bool = PrivateAttr(default=False)

    def validate_downloaded_files(self) -> bool:
        return self._valid

    def download_dataset(self) -> None:
        self._valid = True


class DummyProcessor(BaseRSCDatasetProcessor):
    _valid: bool = PrivateAttr(default=False)

    def validate_dataset_files(self) -> bool:
        return self._valid

    def validate_processed_files(self) -> bool:
        return self._valid

    def process_dataset(self) -> None:
        (self.cache_dir_path / "dummy.txt").write_text("ok")
        self._valid = True


class DummyDataset(BaseRSCDataset):
    downloader_class = DummyDownloader
    processor_class = DummyProcessor


def test_downloader_runs(tmp_path):
    dl = DummyDownloader(name="dummy", outputs_directory=str(tmp_path))
    dl.download()
    assert dl._valid is True


def test_processor_runs(tmp_path):
    proc = DummyProcessor(name="dummy", outputs_directory=str(tmp_path))

    # process should call process_dataset
    proc.process()
    assert (proc.cache_dir_path / "dummy.txt").exists()

    # Second call should skip since validate is True
    assert proc.validate_dataset_files()


def test_dataset_integration(tmp_path):
    dataset = DummyDataset(
        downloader=DummyDownloader(name="dummy", outputs_directory=str(tmp_path)),
        processor=DummyProcessor(name="dummy", outputs_directory=str(tmp_path)),
    )

    # Use BaseRSCDataset convenience methods
    dataset.download()
    dataset.process()

    assert dataset.downloader._valid is True
    assert dataset.processor._valid is True
