from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
import json
import csv
import pathlib
from collections import Counter


CSV_COLUMNS = [
    "sample_id",
    "recording_path",
    "sample_level",
    "t0",
    "t1",
    "split",
    "label",
    "meta",
]

# 학습 쪽(IndexDataset)·프로세서 기반 클래스(BaseRSCDatasetProcessor)가 인덱스 CSV 에 요구하는 열 — split 은 선택(파생 파일로 대체 가능).
REQUIRED_INDEX_COLUMNS = tuple(c for c in CSV_COLUMNS if c != "split")


class SampleRecord(BaseModel):
    sample_id: str
    recording_path: str
    sample_level: Literal["cycle", "clip"]
    t0: Optional[float] = None
    t1: Optional[float] = None
    split: Optional[str] = None
    label: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class SampleIndex(BaseModel):
    records: List[SampleRecord]
    data_root: Optional[str] = None

    def to_csv(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for record in self.records:
                row = {
                    "sample_id": record.sample_id,
                    "recording_path": record.recording_path,
                    "sample_level": record.sample_level,
                    "t0": record.t0,
                    "t1": record.t1,
                    "split": record.split,
                    "label": record.label,
                    "meta": json.dumps(record.meta, sort_keys=True),
                }
                writer.writerow(row)

    def stats(self) -> Dict[str, Any]:
        counts = Counter(r.sample_level for r in self.records)
        label_counts = Counter(r.label for r in self.records if r.label is not None)
        return {
            "total_samples": len(self.records),
            "sample_levels": dict(counts),
            "label_counts": {str(k): v for k, v in label_counts.items()},
        }

    def write_stats(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats(), indent=2, sort_keys=True))

    @classmethod
    def from_csv(cls, path: str | pathlib.Path) -> "SampleIndex":
        path = pathlib.Path(path)
        records: List[SampleRecord] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta = json.loads(row["meta"]) if row.get("meta") else {}
                record = SampleRecord(
                    sample_id=row["sample_id"],
                    recording_path=row["recording_path"],
                    sample_level=row["sample_level"],
                    t0=float(row["t0"]) if row["t0"] else None,
                    t1=float(row["t1"]) if row["t1"] else None,
                    split=row.get("split") or None,
                    label=int(row["label"]) if row["label"] else None,
                    meta=meta,
                )
                records.append(record)
        return cls(records=records)
