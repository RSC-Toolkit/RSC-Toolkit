import pathlib
import logging
import functools


from rsc_toolkit.datasets.base import BaseDatasetDownloader, BaseDataset
from rsc_toolkit.datasets import DatasetFactory
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


class ICBHIDatasetDownloader(BaseDatasetDownloader):
    name: str = "ICBHI"
    outputs_directory: str
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

    def validate_dataset_files(self) -> bool:
        """
        Validate the dataset files.
        """
        if not self.extracted_dir.exists():
            # this is a really simple check. Maybe better to see that all expected files are downloaded and unpacked.
            return False
        return True

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


@DatasetFactory.register_dataset_class("icbhi")
class ICBHIDataset(BaseDataset):
    """
    ICBHI Dataset
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
    """
    downloader: type[ICBHIDatasetDownloader] = ICBHIDatasetDownloader









# def extract_zip(zip_path: str, out_dir: str):
#     with zipfile.ZipFile(zip_path, "r") as z:
#         z.extractall(out_dir)


# @register_dataset("icbhi")
# def download_icbhi(args):

#     out_dir = args.outdir
#     link_to = "data/icbhi_dataset/audio_test_data" # TODO This is temporarily needed.. 

#     """Download and extract the ICBHI dataset."""
#     url = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
#     out_dir = pathlib.Path(out_dir).expanduser().resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     zip_path = out_dir / "ICBHI_final_database.zip"
#     extracted_dir = out_dir / "ICBHI_final_database"

#     print(
#         "Preparing to download ICBHI Dataset.\n" \
#         "Process:\n" \
#         f"\t- Download and unzip single archive from {url}\n" \
#         f"\t- Create symlink to expected data dir: {str(link_to)}"
#         )

#     if not zip_path.exists():
#         print(f"⬇️ Downloading ICBHI dataset to {zip_path}")
#         download_file(url, zip_path)
#     else:
#         print(f"✅ Found existing file at {zip_path}, skipping download.")