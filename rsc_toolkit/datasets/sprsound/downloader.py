import functools
import logging
import pathlib
import shutil

from rsc_toolkit.datasets.base import BaseRSCDatasetDownloader
from rsc_toolkit.datasets.download_utils import download_file, extract_zip


logger = logging.getLogger(__name__)


class SPRSoundDatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "sprsound"
    dataset_download_url: str = "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound/archive/refs/heads/main.zip"

    @functools.cached_property
    def outputs_path(self) -> pathlib.Path:
        path = pathlib.Path(self.outputs_directory).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def zip_path(self) -> pathlib.Path:
        return self.outputs_path / "SPRSound.zip"

    @property
    def extracted_dir(self) -> pathlib.Path:
        return self.outputs_path / "SPRSound-main"

    @property
    def classification_dir(self) -> pathlib.Path:
        return self.outputs_path / "Classification"

    def validate_downloaded_files(self) -> bool:
        if not self.classification_dir.exists():
            return False
        train_dir = self.classification_dir / "train_classification_wav"
        valid_dir = self.classification_dir / "valid_classification_wav"
        if not train_dir.exists() or not valid_dir.exists():
            return False
        if not any(train_dir.glob("*.wav")):
            return False
        if not any(valid_dir.rglob("*.wav")):
            return False
        return True

    def _maybe_flatten_layout(self) -> None:
        if self.classification_dir.exists():
            return

        candidates = [
            self.outputs_path / "SPRSound" / "Classification",
            self.extracted_dir / "Classification",
        ]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            return

        logger.info("Normalizing SPRSound layout: moving %s -> %s", src, self.classification_dir)
        if self.classification_dir.exists():
            if self.force:
                shutil.rmtree(self.classification_dir)
            else:
                return
        shutil.move(str(src), str(self.classification_dir))

        for candidate in (self.outputs_path / "SPRSound", self.extracted_dir):
            try:
                candidate.rmdir()
            except OSError:
                pass

    def download_dataset(self) -> None:
        logger.info(
            "Preparing to download SPRSound Dataset.\n"
            "Process:\n"
            f"\t- Download and unzip archive from {self.dataset_download_url}\n"
        )

        if self.force or not self.zip_path.exists():
            download_file(url=self.dataset_download_url, out_path=str(self.zip_path))
        else:
            logger.info("✅ Found %s, skipping download.", self.zip_path.name)

        if self.force or not self.extracted_dir.exists():
            extract_zip(zip_path=str(self.zip_path), out_dir=str(self.outputs_path))
        else:
            logger.info("✅ Found extracted files at %s, skipping extraction.", self.extracted_dir)

        self._maybe_flatten_layout()
