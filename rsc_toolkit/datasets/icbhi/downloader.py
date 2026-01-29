import pathlib
import logging
import functools
import shutil

from rsc_toolkit.datasets.base import BaseRSCDatasetDownloader
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


class ICBHIDatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "icbhi"
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

    def _has_paired_files(self, root: pathlib.Path) -> bool:
        wav_files = list(root.glob("*.wav"))
        txt_files = list(root.glob("*.txt"))
        if not wav_files or not txt_files:
            return False

        wav_stems = {p.stem for p in wav_files}
        txt_stems = {p.stem for p in txt_files}
        return len(wav_stems & txt_stems) > 0

    def _nested_extracted_dir(self) -> pathlib.Path:
        # Some extractions end up as <outputs>/ICBHI_final_database/ICBHI_final_database/*
        return self.extracted_dir / "ICBHI_final_database"

    def validate_downloaded_files(self) -> bool:
        """
        Validate the dataset files.
        """
        # For the unified layout, consider the dataset valid only when files are at the dataset root.
        return self._has_paired_files(self.outputs_path)

    def _maybe_flatten_extracted_dir(self) -> None:
        """
        Ensure wav/txt pairs live directly under outputs_path.

        The upstream zip contains an `ICBHI_final_database/` folder; we flatten it
        into outputs_path to keep a single unified dataset directory.
        """
        if self._has_paired_files(self.outputs_path):
            return

        candidates = [self.extracted_dir, self._nested_extracted_dir()]
        src = next((p for p in candidates if p.exists() and self._has_paired_files(p)), None)
        if src is None:
            return

        logger.info("Normalizing ICBHI layout: moving files from %s -> %s", src, self.outputs_path)
        for p in src.iterdir():
            dst = self.outputs_path / p.name
            if dst.exists():
                if not self.force:
                    continue
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(p), str(dst))

        # Best-effort cleanup of now-empty directories
        for d in (src, self.extracted_dir, self._nested_extracted_dir()):
            try:
                d.rmdir()
            except OSError:
                pass

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
            # Extract to dataset root to avoid nesting `ICBHI_final_database/ICBHI_final_database/*`.
            extract_zip(zip_path=self.zip_path, out_dir=self.outputs_path)
        else:
            logger.info(f"Found extracted files at {self.extracted_dir}, skipping extraction!")

        self._maybe_flatten_extracted_dir()
