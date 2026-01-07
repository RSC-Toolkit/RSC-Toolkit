import pathlib
import logging
import functools

from rsc_toolkit.datasets.base import BaseRSCDatasetDownloader
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


class ICBHIDatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "ICBHI"
    dataset_download_url: str = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"

    @functools.cached_property
    def outputs_path(self) -> pathlib.Path:
        path = pathlib.Path(self.outputs_directory).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def zip_path(self):
        return self.outputs_path / "ICBHI_final_database.zip"

    @property
    def extracted_dir(self):
        return self.outputs_path / "ICBHI_final_database"

    def validate_downloaded_files(self) -> bool:
        """
        Validate the dataset files.
        """
        if not self.extracted_dir.exists():
            return False

        wav_files = list(self.extracted_dir.glob("*.wav"))
        txt_files = list(self.extracted_dir.glob("*.txt"))
        if not wav_files or not txt_files:
            return False

        wav_stems = {p.stem for p in wav_files}
        txt_stems = {p.stem for p in txt_files}
        paired = wav_stems & txt_stems
        return len(paired) > 0

    def download_dataset(self) -> None:
        """Download and extract the ICBHI dataset."""
        logger.info(
            "Preparing to download ICBHI Dataset.\n" \
            "Process:\n" \
            f"\t- Download and unzip single archive from {self.dataset_download_url}\n"
            )

        if self.force or not self.zip_path.exists():
            download_file(url=self.dataset_download_url, out_path=self.zip_path)
        else:
            logger.info(f"Found dataset at {self.zip_path}, skipping download!")

        if self.force or not self.extracted_dir.exists():
            extract_zip(zip_path=self.zip_path, out_dir=self.extracted_dir)
        else:
            logger.info(f"Found extracted files at {self.extracted_dir}, skipping extraction!")
