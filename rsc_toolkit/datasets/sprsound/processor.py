from __future__ import annotations

from typing import Optional, List, Dict, Literal
import pathlib
import logging
import json

import pandas as pd

from rsc_toolkit.datasets.base import BaseRSCDatasetProcessor
from rsc_toolkit.datasets.sample_index import SampleIndex, SampleRecord

from .processing import (
    DEVICE_TO_ID,
    LOCATION_TO_ID,
    LabelEncoder,
    load_metadata,
    resolve_recording_path,
    normalize_recording_id,
    normalize_split,
    infer_split_from_path,
    _normalize_columns,
    _pick_column,
    coerce_number,
)


logger = logging.getLogger(__name__)


class SPRSoundDatasetProcessor(BaseRSCDatasetProcessor):
    name: str = "sprsound"

    raw_data_directory: Optional[str] = None
    metadata_file: Optional[str] = None

    def validate_processed_files(self) -> bool:
        return False  # TODO: add index validation

    @property
    def data_folder(self) -> pathlib.Path:
        if not self.raw_data_directory:
            raise ValueError("raw_data_directory not set")
        return pathlib.Path(self.raw_data_directory).expanduser().resolve()

    def _resolve_data_root(self) -> pathlib.Path:
        base = self.data_folder
        if base.name.lower() == "classification":
            parent = base.parent
            if (
                (parent / "train_classification_wav").exists()
                or (parent / "valid_classification_wav").exists()
                or (parent / "train_classification_json").exists()
                or (parent / "valid_classification_json").exists()
            ):
                return parent.resolve()
        if (
            (base / "train_classification_wav").exists()
            or (base / "valid_classification_wav").exists()
            or (base / "train_classification_json").exists()
            or (base / "valid_classification_json").exists()
        ):
            return base.resolve()

        if (base / "Classification").exists():
            return (base / "Classification").resolve()

        for candidate in ("SPRSound", "SPRSound-main"):
            if (base / candidate / "Classification").exists():
                return (base / candidate / "Classification").resolve()

        return base.resolve()

    def _package_data_dir(self) -> pathlib.Path:
        return (pathlib.Path(__file__).parent / "data" / "metadata.csv").resolve()

    def _get_metadata_path(self) -> pathlib.Path:
        if self.metadata_file:
            return pathlib.Path(self.metadata_file).expanduser().resolve()
        data_root = self._resolve_data_root()
        primary = (data_root / "data" / "metadata.csv").resolve()
        if primary.exists():
            return primary
        return self._package_data_dir().resolve()

    def _write_split_file(self, split_map: Dict[str, str]) -> Optional[pathlib.Path]:
        if not split_map:
            return None
        path = self.index_dir_path / "official_split.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{rid} {split}" for rid, split in sorted(split_map.items())]
        path.write_text("\n".join(lines) + "\n")
        return path

    def _write_label_map(self, label_encoder: LabelEncoder) -> Optional[pathlib.Path]:
        if not label_encoder.label_to_id:
            return None
        path = self.index_dir_path / "label_map.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(label_encoder.label_to_id, indent=2, sort_keys=True))
        return path

    def _build_meta(
        self,
        *,
        row,
        patient_col: Optional[str],
        age_col: Optional[str],
        gender_col: Optional[str],
        location_col: Optional[str],
        record_num_col: Optional[str],
        record_ann_col: Optional[str],
        split_col: Optional[str],
    ) -> Dict[str, object]:
        meta: Dict[str, object] = {}

        if patient_col:
            meta["patient_id"] = str(row[patient_col])
        if age_col:
            meta["age"] = coerce_number(row[age_col])
        if gender_col:
            meta["gender"] = coerce_number(row[gender_col])
        if location_col:
            location = str(row[location_col]).strip()
            if location:
                meta["recording_location"] = location
                meta["location_id"] = LOCATION_TO_ID.get(location)
        if record_num_col:
            meta["recording_number"] = coerce_number(row[record_num_col])
        if record_ann_col:
            meta["record_annotation"] = str(row[record_ann_col]).strip()
        if split_col:
            raw_split = str(row[split_col]).strip().lower()
            if raw_split:
                meta["split_raw"] = raw_split
                meta["split"] = normalize_split(raw_split)

        meta["device"] = "Yunting2"
        meta["device_id"] = DEVICE_TO_ID.get("Yunting2")
        return meta

    def _make_cycle_sample_id(self, recording_id: str, t0: float, t1: float) -> str:
        return f"{recording_id}_cycle_{t0:.3f}_{t1:.3f}"

    def _build_cycle_index(
        self,
        *,
        df: pd.DataFrame,
        data_root: pathlib.Path,
        label_encoder: LabelEncoder,
    ) -> SampleIndex:
        cols = _normalize_columns(df)

        filename_col = _pick_column(cols, ["filename", "file", "filepath", "path", "audio", "wav"])
        if not filename_col:
            raise ValueError("metadata.csv missing a filename/path column")

        split_col = _pick_column(cols, ["split", "set", "subset", "partition", "fold"])
        label_col = _pick_column(cols, ["label", "class", "diagnosis", "type", "target", "y"])
        start_col = _pick_column(cols, ["start", "begin", "t0", "onset"])
        end_col = _pick_column(cols, ["end", "stop", "t1", "offset"])

        patient_col = _pick_column(cols, ["patient_number", "patient_id", "patient"])
        age_col = _pick_column(cols, ["age"])
        gender_col = _pick_column(cols, ["gender", "sex"])
        location_col = _pick_column(cols, ["recording_location", "location", "site"])
        record_num_col = _pick_column(cols, ["recording_number", "record_number"])
        record_ann_col = _pick_column(cols, ["record_annotation", "recording_annotation", "annotation"])

        if start_col is None or end_col is None:
            raise ValueError("metadata.csv missing start/end columns for cycle index")

        records: List[SampleRecord] = []

        split_map: Dict[str, str] = {}
        for _, row in df.iterrows():
            filename = str(row[filename_col])
            raw_split = row[split_col] if split_col else None
            split = normalize_split(raw_split)
            if split is None:
                split = infer_split_from_path(filename)
            recording_path = resolve_recording_path(
                data_root=data_root,
                filename=filename,
                split=split,
            )
            if recording_path is None or not recording_path.exists():
                logger.warning(
                    f"{data_root} {filename} {split}"
                )
                logger.warning("Skipping missing audio file: %s", recording_path)
                continue

            try:
                recording_path_str = str(recording_path.relative_to(data_root))
            except ValueError:
                recording_path_str = str(recording_path)

            recording_id = normalize_recording_id(recording_path.name)

            t0 = float(row[start_col])
            t1 = float(row[end_col])
            label = label_encoder.encode(row[label_col]) if label_col else None

            meta = self._build_meta(
                row=row,
                patient_col=patient_col,
                age_col=age_col,
                gender_col=gender_col,
                location_col=location_col,
                record_num_col=record_num_col,
                record_ann_col=record_ann_col,
                split_col=split_col,
            )
            meta["recording_id"] = recording_id

            if split:
                split_map.setdefault(recording_id, split)

            records.append(
                SampleRecord(
                    sample_id=self._make_cycle_sample_id(recording_id, t0, t1),
                    recording_path=recording_path_str,
                    sample_level="cycle",
                    t0=t0,
                    t1=t1,
                    split=split,
                    label=label,
                    meta=meta,
                )
            )

        self._write_split_file(split_map)
        return SampleIndex(records=records, data_root=str(data_root))

    def _build_clip_index(
        self,
        *,
        df: pd.DataFrame,
        data_root: pathlib.Path,
        label_encoder: LabelEncoder,
    ) -> SampleIndex:
        cols = _normalize_columns(df)
        filename_col = _pick_column(cols, ["filename", "file", "filepath", "path", "audio", "wav"])
        if not filename_col:
            raise ValueError("metadata.csv missing a filename/path column")

        split_col = _pick_column(cols, ["split", "set", "subset", "partition", "fold"])
        record_ann_col = _pick_column(cols, ["record_annotation", "recording_annotation", "annotation"])

        patient_col = _pick_column(cols, ["patient_number", "patient_id", "patient"])
        age_col = _pick_column(cols, ["age"])
        gender_col = _pick_column(cols, ["gender", "sex"])
        location_col = _pick_column(cols, ["recording_location", "location", "site"])
        record_num_col = _pick_column(cols, ["recording_number", "record_number"])

        records: List[SampleRecord] = []

        for filename, group in df.groupby(filename_col):
            row = group.iloc[0]
            raw_split = row[split_col] if split_col else None
            split = normalize_split(raw_split)
            if split is None:
                split = infer_split_from_path(str(filename))

            recording_path = resolve_recording_path(
                data_root=data_root,
                filename=str(filename),
                split=split,
            )
            if recording_path is None or not recording_path.exists():
                logger.warning("Skipping missing audio file: %s", filename)
                continue

            try:
                recording_path_str = str(recording_path.relative_to(data_root))
            except ValueError:
                recording_path_str = str(recording_path)

            recording_id = normalize_recording_id(recording_path.name)

            label = None
            if record_ann_col:
                label = label_encoder.encode(row[record_ann_col])

            meta = self._build_meta(
                row=row,
                patient_col=patient_col,
                age_col=age_col,
                gender_col=gender_col,
                location_col=location_col,
                record_num_col=record_num_col,
                record_ann_col=record_ann_col,
                split_col=split_col,
            )
            meta["recording_id"] = recording_id

            records.append(
                SampleRecord(
                    sample_id=recording_id,
                    recording_path=recording_path_str,
                    sample_level="clip",
                    t0=None,
                    t1=None,
                    split=split,
                    label=label,
                    meta=meta,
                )
            )

        return SampleIndex(records=records, data_root=str(data_root))

    def process_dataset(self) -> None:
        metadata_path = self._get_metadata_path()
        df = load_metadata(metadata_path)
        data_root = self._resolve_data_root().resolve()

        label_encoder = LabelEncoder()

        cycle_index = self._build_cycle_index(
            df=df,
            data_root=data_root,
            label_encoder=label_encoder,
        )
        self.index_dir_path.mkdir(parents=True, exist_ok=True)
        cycle_path = self.index_dir_path / "index_cycle.csv"
        cycle_stats = self.index_dir_path / "index_cycle_stats.json"
        cycle_index.to_csv(cycle_path)
        cycle_index.write_stats(cycle_stats)
        logger.info("Wrote cycle index to %s", cycle_path)

        clip_index = self._build_clip_index(
            df=df,
            data_root=data_root,
            label_encoder=label_encoder,
        )
        clip_path = self.index_dir_path / "index_clip.csv"
        clip_stats = self.index_dir_path / "index_clip_stats.json"
        clip_index.to_csv(clip_path)
        clip_index.write_stats(clip_stats)
        logger.info("Wrote clip index to %s", clip_path)

        self._write_label_map(label_encoder)

    def build_index(self, sample_level: Literal["cycle", "clip"] = "cycle") -> SampleIndex:
        metadata_path = self._get_metadata_path()
        df = load_metadata(metadata_path)
        data_root = self._resolve_data_root().resolve()

        label_encoder = LabelEncoder()
        if sample_level == "cycle":
            index = self._build_cycle_index(
                df=df,
                data_root=data_root,
                label_encoder=label_encoder,
            )
        else:
            index = self._build_clip_index(
                df=df,
                data_root=data_root,
                label_encoder=label_encoder,
            )

        self.index_dir_path.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir_path / f"index_{sample_level}.csv"
        stats_path = self.index_dir_path / f"index_{sample_level}_stats.json"
        index.to_csv(index_path)
        index.write_stats(stats_path)
        logger.info("Wrote index to %s", index_path)
        self._write_label_map(label_encoder)
        return index
