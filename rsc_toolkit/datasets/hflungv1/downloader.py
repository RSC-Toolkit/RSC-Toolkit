from __future__ import annotations

import functools
import logging
import pathlib
import shutil
import subprocess
from typing import Iterable, List

from rsc_toolkit.datasets.base import BaseRSCDatasetDownloader
from rsc_toolkit.datasets.download_utils import download_file


logger = logging.getLogger(__name__)


class HFLungV1DatasetDownloader(BaseRSCDatasetDownloader):
    name: str = "hflungv1"
    base_url: str = "https://gitlab.com/techsupportHF/HF_Lung_V1/-/raw/master"

    @functools.cached_property
    def outputs_path(self) -> pathlib.Path:
        path = pathlib.Path(self.outputs_directory).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def metadata_path(self) -> pathlib.Path:
        return self.outputs_path / "metadata.csv"

    @property
    def train_dir(self) -> pathlib.Path:
        return self.outputs_path / "train"

    @property
    def test_dir(self) -> pathlib.Path:
        return self.outputs_path / "test"

    @staticmethod
    def _train_parts() -> List[str]:
        return [f"train.7z.{idx:03d}" for idx in range(1, 11)]

    @staticmethod
    def _test_parts() -> List[str]:
        return [f"test.7z.{idx:03d}" for idx in range(1, 4)]

    def validate_downloaded_files(self) -> bool:
        if not self.metadata_path.exists():
            return False
        if not self.train_dir.exists() or not self.test_dir.exists():
            return False
        if not any(self.train_dir.glob("*.wav")):
            return False
        if not any(self.test_dir.glob("*.wav")):
            return False
        return True

    def _download_parts(self, parts: Iterable[str]) -> List[pathlib.Path]:
        local_files: List[pathlib.Path] = []
        for fname in parts:
            url = f"{self.base_url}/{fname}"
            dest = self.outputs_path / fname
            local_files.append(dest)
            if self.force or not dest.exists():
                logger.info("⬇️ Downloading %s", fname)
                download_file(url, str(dest))
            else:
                logger.info("✅ Found %s, skipping.", fname)
        return local_files

    def _extract_7z(self, archive_path: pathlib.Path) -> None:
        logger.info("📦 Extracting %s into %s", archive_path.name, self.outputs_path)
        try:
            subprocess.run(
                ["7z", "x", str(archive_path), f"-o{self.outputs_path}"],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("7z command not found. Please install 7-Zip / p7zip.") from exc

    def _maybe_flatten_layout(self) -> None:
        """
        Normalize layout so train/test/metadata.csv live directly in outputs_path.
        """
        if self.metadata_path.exists() and self.train_dir.exists() and self.test_dir.exists():
            return

        nested = self.outputs_path / "HF_Lung_V1"
        if not nested.exists():
            return

        for name in ("train", "test", "metadata.csv"):
            src = nested / name
            dst = self.outputs_path / name
            if not src.exists():
                continue
            if dst.exists():
                if not self.force:
                    continue
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))

        try:
            nested.rmdir()
        except OSError:
            pass

    def _package_metadata_path(self) -> pathlib.Path:
        return (pathlib.Path(__file__).parent / "data" / "metadata.csv").resolve()

    def _ensure_metadata(self) -> None:
        # The upstream GitLab repository has no metadata.csv, so the packaged copy (generated from the label CSVs)
        # is copied into the data folder to satisfy the validate contract.
        if self.metadata_path.exists():
            return
        bundled = self._package_metadata_path()
        if not bundled.exists():
            logger.warning("Bundled metadata.csv not found at %s", bundled)
            return
        logger.info("Copying bundled metadata.csv -> %s", self.metadata_path)
        shutil.copyfile(bundled, self.metadata_path)

    def download_dataset(self) -> None:
        logger.info(
            "Preparing to download HF_Lung_V1.\n"
            "Process:\n"
            f"\t- Download split 7z archives from {self.base_url}\n"
            "\t- Extract train and test archives\n"
        )

        train_parts = self._download_parts(self._train_parts())
        test_parts = self._download_parts(self._test_parts())

        if self.force or not self.train_dir.exists():
            self._extract_7z(train_parts[0])
        else:
            logger.info("✅ Found train folder, skipping extraction.")

        if self.force or not self.test_dir.exists():
            self._extract_7z(test_parts[0])
        else:
            logger.info("✅ Found test folder, skipping extraction.")

        self._maybe_flatten_layout()
        self._ensure_metadata()
