# rsc_toolkit/datasets/icbhi/processor.py
from __future__ import annotations

from typing import Optional, List, Literal
import pathlib
import logging
import torch
from torch.utils.data import DataLoader

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
        return index

    # def preprocess_features_from_index(
    #     self,
    #     index_csv: str | pathlib.Path,
    #     *,
    #     sample_level: str = "cycle",
    #     batch_size: int = 1,
    #     num_workers: int = 0,
    #     fade: bool = True,
    #     cache_dir: str | pathlib.Path | None = None,
    # ):
    #     if not self.feature_processor:
    #         raise RuntimeError("preprocess_features=True but feature_processor is None")

    #     index_csv = pathlib.Path(index_csv).expanduser().resolve()

    #     ds = ICBHIIndexDataset(
    #         index_csv=index_csv,
    #         data_root=self.data_folder,
    #         sampling_rate=self.sampling_rate,
    #         sample_level=sample_level,
    #         return_meta=True,
    #         fade=fade,
    #     )

    #     loader = DataLoader(
    #         ds,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=False,
    #         persistent_workers=False,
    #     )

    #     # Ensure processor cache is set up
    #     if cache_dir is None:
    #         cache_dir = pathlib.Path(self.outputs_directory) / "features"
    #     cache_dir = pathlib.Path(cache_dir)
    #     if not self.feature_processor.cache_features or self.feature_processor.cache_dir != str(cache_dir):
    #         self.feature_processor.cache_features = True
    #         self.feature_processor.cache_dir = str(cache_dir)
    #         # re-init cache with new dir/config
    #         self.feature_processor._init_cache()

    #     fp = self.feature_processor

    #     for batch in loader:
    #         # batch is a dict of tensors/lists
    #         audio = batch["audio"]            # shape: [B, C, T]
    #         sample_ids = batch["sample_id"]   # list[str] or tuple[str]
    #         # Compute + cache features per sample_id
    #         if hasattr(fp, "make_features_batch"):
    #             # Optional batch path if processor supports it (not cache-aware internally)
    #             feats_list = fp.make_features_batch(audio, sample_ids=sample_ids)  # type: ignore[arg-type]
    #             cache = getattr(fp, "_feature_cache", None)
    #             for sid, feats in zip(sample_ids, feats_list):
    #                 if cache:
    #                     cache.put(sid, feats)
    #                 else:
    #                     # fallback: cache disabled, no-op aside from compute
    #                     pass
    #         else:
    #             for i in range(audio.shape[0]):
    #                 sid = sample_ids[i]
    #                 x = audio[i]  # [C, T]
    #                 fp.make_features_cached(x, sample_id=sid)

    #         # Explicit cleanup (optional but fine)
    #         del batch
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
