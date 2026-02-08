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
    find_annotation_path,
    parse_cycle_annotations,
    normalize_recording_id,
    _normalize_columns,
    _pick_column,
)


logger = logging.getLogger(__name__)


class HFLungV1DatasetProcessor(BaseRSCDatasetProcessor):
    name: str = "hflungv1"

    # Raw unpacked data directory containing train/ test/ metadata.csv
    raw_data_directory: Optional[str] = None
    metadata_file: Optional[str] = None
    labels_files: Optional[List[str]] = None

    def validate_processed_files(self) -> bool:
        return False  # TODO: add index validation

    @property
    def data_folder(self) -> pathlib.Path:
        if not self.raw_data_directory:
            raise ValueError("raw_data_directory not set")
        return pathlib.Path(self.raw_data_directory).expanduser().resolve()

    def _resolve_data_root(self) -> pathlib.Path:
        base = self.data_folder
        candidate = base / "data"
        if (
            (base / "metadata.csv").exists()
            or (base / "train").exists()
            or (base / "test").exists()
        ):
            return base
        if (
            candidate.exists()
            and (
                (candidate / "metadata.csv").exists()
                or (candidate / "train").exists()
                or (candidate / "test").exists()
            )
        ):
            return candidate.resolve()
        return base

    def _package_data_dir(self) -> pathlib.Path:
        return (pathlib.Path(__file__).parent / "data").resolve()

    def _get_metadata_path(self) -> pathlib.Path:
        if self.metadata_file:
            return pathlib.Path(self.metadata_file).expanduser().resolve()
        data_root = self._resolve_data_root()
        primary = (data_root / "metadata.csv").resolve()
        if primary.exists():
            return primary
        fallback = self._package_data_dir() / "metadata.csv"
        return fallback.resolve()

    def _resolve_labels_files(self, metadata_path: pathlib.Path) -> List[pathlib.Path]:
        if self.labels_files:
            return [pathlib.Path(p).expanduser().resolve() for p in self.labels_files]

        labels_dir = metadata_path.parent
        intra = [
            labels_dir / "train_intra_segmented_labels.csv",
            labels_dir / "test_intra_segmented_labels.csv",
        ]
        unified = [
            labels_dir / "train_unified_segmented_labels.csv",
            labels_dir / "test_unified_segmented_labels.csv",
        ]

        intra_existing = [p.resolve() for p in intra if p.exists()]
        if intra_existing:
            return intra_existing

        return [p.resolve() for p in unified if p.exists()]

    def _load_segmented_labels(self, metadata_path: pathlib.Path) -> Optional[pd.DataFrame]:
        label_files = self._resolve_labels_files(metadata_path)
        if not label_files:
            return None

        dfs: List[pd.DataFrame] = []
        for path in label_files:
            df = pd.read_csv(path)
            if "split" not in df.columns:
                stem = path.stem.lower()
                split = None
                if "train" in stem:
                    split = "train"
                elif "test" in stem:
                    split = "test"
                if split:
                    df = df.copy()
                    df["split"] = split
            dfs.append(df)
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def _build_split_map(self, df, filename_col: str, split_col: Optional[str]) -> Dict[str, str]:
        split_map: Dict[str, str] = {}
        if split_col is None:
            return split_map

        for _, row in df.iterrows():
            filename = str(row[filename_col])
            split = row[split_col]
            if split is None:
                continue
            split_str = str(split).strip().lower()
            if not split_str:
                continue
            recording_id = pathlib.Path(filename).stem
            split_map[recording_id] = split_str
        return split_map

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
        device_col: Optional[str],
        location_col: Optional[str],
        split_col: Optional[str],
    ) -> Dict[str, object]:
        meta: Dict[str, object] = {}

        if device_col:
            device = str(row[device_col]).strip()
            meta["device"] = device
            meta["device_id"] = DEVICE_TO_ID.get(device)
        if location_col:
            location = str(row[location_col]).strip()
            meta["location"] = location
            meta["location_id"] = LOCATION_TO_ID.get(location)
        if split_col:
            split = str(row[split_col]).strip().lower()
            if split:
                meta["split"] = split
        return meta

    def _make_cycle_sample_id(self, recording_id: str, t0: float, t1: float) -> str:
        return f"{recording_id}_cycle_{t0:.3f}_{t1:.3f}"

    def _build_index_for_level(
        self,
        *,
        df,
        data_root: pathlib.Path,
        sample_level: Literal["cycle", "clip"],
    ) -> SampleIndex:
        cols = _normalize_columns(df)

        filename_col = _pick_column(
            cols,
            [
                "filename",
                "file",
                "filepath",
                "path",
                "audio",
                "audio_path",
                "recording",
                "recording_path",
                "wav",
            ],
        )
        if not filename_col:
            raise ValueError("metadata.csv missing a filename/path column")

        split_col = _pick_column(cols, ["split", "set", "subset", "partition", "fold"])
        label_col = _pick_column(cols, ["label", "class", "diagnosis", "disease", "target", "y"])
        device_col = _pick_column(cols, ["device", "stethoscope", "device_type"])
        location_col = _pick_column(cols, ["location", "chest_location", "lung_location", "site"])

        label_encoder = LabelEncoder()
        records: List[SampleRecord] = []

        metadata_map: Dict[str, Dict[str, object]] = {}
        for _, row in df.iterrows():
            key = normalize_recording_id(str(row[filename_col]))
            meta = self._build_meta(
                row=row,
                device_col=device_col,
                location_col=location_col,
                split_col=split_col,
            )
            metadata_map[key] = meta

        labels_df = None
        if sample_level == "cycle":
            labels_df = self._load_segmented_labels(self._get_metadata_path())

        if labels_df is not None and sample_level == "cycle":
            labels_cols = _normalize_columns(labels_df)
            labels_filename_col = _pick_column(labels_cols, ["filename", "file", "filepath", "path", "audio"])
            if not labels_filename_col:
                raise ValueError("segmented labels missing a filename column")
            labels_label_col = _pick_column(labels_cols, ["final", "label", "class", "diagnosis", "target", "y"])
            labels_split_col = _pick_column(labels_cols, ["split", "set", "subset", "partition", "fold"])
            start_col = _pick_column(labels_cols, ["start", "begin", "t0", "onset"])
            end_col = _pick_column(labels_cols, ["end", "stop", "t1", "offset"])

            if start_col is None or end_col is None:
                raise ValueError("segmented labels missing start/end columns")

            for _, row in labels_df.iterrows():
                filename = str(row[labels_filename_col])
                split = row[labels_split_col] if labels_split_col else None
                recording_path = resolve_recording_path(
                    data_root=data_root,
                    filename=filename,
                    split=str(split).strip().lower() if split is not None else None,
                )
                if recording_path is None or not recording_path.exists():
                    logger.warning("Skipping missing audio file: %s", filename)
                    continue

                try:
                    recording_path_str = str(recording_path.relative_to(data_root))
                except ValueError:
                    recording_path_str = str(recording_path)

                recording_id = normalize_recording_id(filename)
            meta = dict(metadata_map.get(recording_id, {}))
            split_val = None
            if labels_split_col and split is not None:
                split_val = str(split).strip().lower()
                meta.setdefault("split", split_val)
            elif "split" in meta:
                split_val = str(meta["split"]).strip().lower()
            meta["recording_id"] = recording_path.stem

            label = label_encoder.encode(row[labels_label_col]) if labels_label_col else None
            t0 = float(row[start_col])
            t1 = float(row[end_col])

                records.append(
                    SampleRecord(
                        sample_id=self._make_cycle_sample_id(recording_path.stem, t0, t1),
                        recording_path=recording_path_str,
                        sample_level="cycle",
                        t0=t0,
                        t1=t1,
                        split=split_val,
                        label=label,
                        meta=meta,
                    )
                )
        else:
            for _, row in df.iterrows():
                filename = str(row[filename_col])
                split = row[split_col] if split_col else None
                recording_path = resolve_recording_path(
                    data_root=data_root,
                    filename=filename,
                    split=str(split).strip().lower() if split is not None else None,
                )
                if recording_path is None or not recording_path.exists():
                    logger.warning("Skipping missing audio file: %s", filename)
                    continue

                try:
                    recording_path_str = str(recording_path.relative_to(data_root))
                except ValueError:
                    recording_path_str = str(recording_path)

                label = label_encoder.encode(row[label_col]) if label_col else None
                split_val = str(split).strip().lower() if split is not None else None
                meta = self._build_meta(
                    row=row,
                    device_col=device_col,
                    location_col=location_col,
                    split_col=split_col,
                )
                meta["recording_id"] = recording_path.stem

                if sample_level == "clip":
                    records.append(
                        SampleRecord(
                            sample_id=recording_path.stem,
                            recording_path=recording_path_str,
                            sample_level="clip",
                            t0=None,
                            t1=None,
                            split=split_val,
                            label=label,
                            meta=meta,
                        )
                    )
                    continue

                annotation_path = find_annotation_path(recording_path)
                if annotation_path is None:
                    logger.warning("Missing annotations for %s; skipping cycle index.", recording_path.name)
                    continue

                cycles = parse_cycle_annotations(
                    annotation_path,
                    label_encoder=label_encoder,
                    default_label=label,
                )
                if not cycles:
                    logger.warning("No cycles parsed for %s", recording_path.name)
                    continue

                for t0, t1, cycle_label in cycles:
                    records.append(
                        SampleRecord(
                            sample_id=self._make_cycle_sample_id(recording_path.stem, t0, t1),
                            recording_path=recording_path_str,
                            sample_level="cycle",
                            t0=t0,
                            t1=t1,
                            split=split_val,
                            label=cycle_label,
                            meta=meta,
                        )
                    )

        self._write_label_map(label_encoder)
        split_map = self._build_split_map(df, filename_col=filename_col, split_col=split_col)
        self._write_split_file(split_map)

        return SampleIndex(records=records, data_root=str(data_root))

    def process_dataset(self) -> None:
        metadata_path = self._get_metadata_path()
        df = load_metadata(metadata_path)
        data_root = self._resolve_data_root().resolve()

        for level in ("cycle", "clip"):
            index = self._build_index_for_level(df=df, data_root=data_root, sample_level=level)  # type: ignore[arg-type]
            self.index_dir_path.mkdir(parents=True, exist_ok=True)
            index_path = self.index_dir_path / f"index_{level}.csv"
            stats_path = self.index_dir_path / f"index_{level}_stats.json"
            index.to_csv(index_path)
            index.write_stats(stats_path)
            logger.info("Wrote %s index to %s", level, index_path)

    def build_index(self, sample_level: Literal["cycle", "clip"] = "cycle") -> SampleIndex:
        metadata_path = self._get_metadata_path()
        df = load_metadata(metadata_path)
        data_root = self._resolve_data_root().resolve()

        index = self._build_index_for_level(df=df, data_root=data_root, sample_level=sample_level)
        self.index_dir_path.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir_path / f"index_{sample_level}.csv"
        stats_path = self.index_dir_path / f"index_{sample_level}_stats.json"
        index.to_csv(index_path)
        index.write_stats(stats_path)
        logger.info("Wrote index to %s", index_path)
        return index
