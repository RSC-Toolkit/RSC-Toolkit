# rsc_toolkit/datasets/icbhi/processor.py
from __future__ import annotations

from typing import Optional, List, Literal
import shutil
import pathlib
import logging

from rsc_toolkit.datasets.base import BaseRSCDatasetProcessor
from rsc_toolkit.datasets.sample_index import SampleIndex, SampleRecord

from .models import RecordingRef
from .processing import (
    discover_recordings,
    load_patient_metadata,
    attach_patient_metadata,
    parse_cycle_annotations,
    make_cycle_sample_id,
    build_meta_dict,
    try_relpath,
)

logger = logging.getLogger(__name__)


class ICBHIDatasetProcessor(BaseRSCDatasetProcessor):
    """
    Processor responsibilities:
      - discover raw files
      - attach metadata
      - build and write indices (cycle + clip)
      - optionally run feature preprocessing (ideally driven by the index)
    """

    name: str = "icbhi"
    sampling_rate: int = 10000

    # Raw unpacked data directory (folder containing .wav + .txt pairs)
    raw_data_directory: Optional[str] = None

    # Metadata file (optional)
    metadata_file: Optional[str] = None  # if None, defaults to package data/metadata.txt

    @property
    def data_folder(self) -> pathlib.Path:
        if not self.raw_data_directory:
            raise ValueError("raw_data_directory not set")
        return pathlib.Path(self.raw_data_directory).expanduser().resolve()

    @property
    def _default_metadata_path(self) -> pathlib.Path:
        # same pattern you used: package-local data/metadata.txt
        return (pathlib.Path(__file__).parent / "data" / "metadata.txt").resolve()

    def _get_metadata_path(self) -> pathlib.Path:
        if self.metadata_file:
            return pathlib.Path(self.metadata_file).expanduser().resolve()
        return self._default_metadata_path

    @property
    def _packaged_split_path(self) -> pathlib.Path:
        return (pathlib.Path(__file__).parent / "data" / "official_split.txt").resolve()

    def _write_split_file(self) -> Optional[pathlib.Path]:
        """
        Copy the packaged official split next to the index.

        ICBHI indexes do not fill the split column (the official split is per recording, so
        a separate file is closer to the original than per-row values). The sprsound and hflungv1
        processors write official_split.txt to index_dir as well, and
        IndexDataset looks for it there; without it
        `IndexDataset(csv, split="test")` has nothing to filter on.
        """
        src = self._packaged_split_path
        if not src.exists():
            logger.warning("Packaged official_split.txt not found: %s", src)
            return None
        dst = self.index_dir_path / "official_split.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        logger.info("Wrote official split to %s", dst)
        return dst

    def _discover_recordings(self) -> List[RecordingRef]:
        return discover_recordings(self.data_folder)

    def _attach_metadata_if_available(self, recordings: List[RecordingRef]) -> None:
        meta_path = self._get_metadata_path()
        try:
            df = load_patient_metadata(meta_path)
        except FileNotFoundError:
            logger.warning("Metadata file not found (%s); continuing without patient metadata", meta_path)
            return
        except Exception as e:
            logger.warning("Failed to load metadata (%s): %s; continuing without patient metadata", meta_path, e)
            return

        attach_patient_metadata(recordings, df)

    def _build_index_for_recordings(
        self,
        recordings: List[RecordingRef],
        sample_level: Literal["cycle", "clip"],
    ) -> SampleIndex:
        records: List[SampleRecord] = []
        data_root = self.data_folder.resolve()

        for rec in recordings:
            meta = build_meta_dict(rec)
            recording_path = try_relpath(rec.wav_path, data_root)

            if sample_level == "clip":
                records.append(
                    SampleRecord(
                        sample_id=rec.recording_id,
                        recording_path=recording_path,
                        sample_level="clip",
                        t0=None,
                        t1=None,
                        label=None,
                        meta=meta,
                    )
                )
                continue

            # cycle-level
            ann = parse_cycle_annotations(rec.txt_path)
            for row in ann:
                sid = make_cycle_sample_id(rec.recording_id, row.start, row.end)
                records.append(
                    SampleRecord(
                        sample_id=sid,
                        recording_path=recording_path,
                        sample_level="cycle",
                        t0=row.start,
                        t1=row.end,
                        label=row.label,
                        meta=meta,
                    )
                )

        return SampleIndex(records=records, data_root=str(data_root))

    def process_dataset(self) -> None:
        recordings = self._discover_recordings()
        logger.info("Found %d recordings", len(recordings))
        if not recordings:
            logger.warning("No recordings found in %s", self.data_folder)
            return

        self._attach_metadata_if_available(recordings)

        # Build + write both indices (cheap, deterministic, reusable)
        for level in ("cycle", "clip"):
            index = self._build_index_for_recordings(recordings, sample_level=level)  # type: ignore[arg-type]
            self.index_dir_path.mkdir(parents=True, exist_ok=True)
            index_path = self.index_dir_path / f"index_{level}.csv"
            stats_path = self.index_dir_path / f"index_{level}_stats.json"
            index.to_csv(index_path)
            index.write_stats(stats_path)
            logger.info("Wrote %s index to %s", level, index_path)

        self._write_split_file()


    def build_index(self, sample_level: Literal["cycle", "clip"] = "cycle") -> SampleIndex:
        recordings = self._discover_recordings()
        logger.info("Found %d recordings for index", len(recordings))
        if not recordings:
            return SampleIndex(records=[], data_root=str(self.data_folder))

        self._attach_metadata_if_available(recordings)

        index = self._build_index_for_recordings(recordings, sample_level=sample_level)
        self.index_dir_path.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir_path / f"index_{sample_level}.csv"
        stats_path = self.index_dir_path / f"index_{sample_level}_stats.json"
        index.to_csv(index_path)
        index.write_stats(stats_path)
        logger.info("Wrote index to %s", index_path)
        self._write_split_file()
        return index
