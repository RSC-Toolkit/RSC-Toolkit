# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import pathlib

from pydantic import BaseModel, ConfigDict, field_validator


DEVICE_IDS: Dict[str, int] = {
    "Meditron": 0,
    "LittC2SE": 1,
    "Litt3200": 2,
    "AKGC417L": 3,
}

CHEST_LOCATIONS: Dict[str, str] = {
    "Tc": "Trachea",
    "Al": "Anterior left",
    "Ar": "Anterior right",
    "Pl": "Posterior left",
    "Pr": "Posterior right",
    "Ll": "Lateral left",
    "Lr": "Lateral right",
}


class RecordingMetadata(BaseModel):
    """
    Metadata for an ICBHI recording.

    Note: This model is intentionally "data-only":
    - No file I/O
    - No torchaudio
    - No pandas
    """

    model_config = ConfigDict(extra="forbid")

    # From filename
    patient_id: int
    recording_index: str
    chest_location: str
    acquisition_mode: str
    device: str
    ext: str

    # From metadata.txt (optional)
    age: Optional[float] = None
    sex: Optional[int] = None  # 0=Male, 1=Female
    adult_BMI: Optional[float] = None
    child_weight: Optional[float] = None
    child_height: Optional[float] = None
    chest_location_id: Optional[int] = None

    @field_validator("patient_id", mode="before")
    @classmethod
    def _patient_id_to_int(cls, v):
        return int(v) if isinstance(v, str) else v

    @field_validator("chest_location", mode="before")
    @classmethod
    def _normalize_chest_location(cls, v):
        return CHEST_LOCATIONS.get(v, v)

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in DEVICE_IDS:
            raise ValueError(f"Unknown device: {v}")
        return v

    @property
    def device_id(self) -> int:
        return DEVICE_IDS[self.device]

    @classmethod
    def from_wav_path(cls, wav_path: pathlib.Path) -> "RecordingMetadata":
        """
        Parse ICBHI filename:
        <patient_id>_<rec_idx>_<chest>_<acq>_<device>.wav
        """
        parts = wav_path.stem.split("_")
        if len(parts) != 5:
            raise ValueError(f"Unexpected filename format: {wav_path.name}")
        patient_id, rec_idx, chest, acq, device = parts
        return cls(
            patient_id=patient_id,
            recording_index=rec_idx,
            chest_location=chest,
            acquisition_mode=acq,
            device=device,
            ext=wav_path.suffix,
        )


class RecordingRef(BaseModel):
    """
    A lightweight pointer to raw files + parsed metadata.
    No loading/processing here.
    """

    model_config = ConfigDict(extra="forbid")

    recording_id: str  # stable identifier, usually wav stem
    wav_path: pathlib.Path
    txt_path: pathlib.Path
    meta: RecordingMetadata

    @classmethod
    def from_paths(cls, wav_path: pathlib.Path, txt_path: pathlib.Path) -> "RecordingRef":
        meta = RecordingMetadata.from_wav_path(wav_path)
        return cls(
            recording_id=wav_path.stem,
            wav_path=wav_path,
            txt_path=txt_path,
            meta=meta,
        )


class CycleRef(BaseModel):
    """
    Pointer to a cycle segment within a recording.

    This is the unit you want in your SampleIndex:
    it references where the audio lives, plus the segment boundaries and label.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    recording_id: str
    recording_path: str  # relative to dataset root if possible
    sample_level: Literal["cycle"] = "cycle"

    t0: float
    t1: float
    label: Optional[int] = None  # crackle + 2*wheeze etc.

    # Minimal optional fields extracted from annotation row
    crackle: Optional[int] = None
    wheeze: Optional[int] = None

    # Denormalized metadata (helps downstream consumers avoid joins)
    meta: Dict[str, Any] = {}


class ClipRef(BaseModel):
    """
    Pointer to a full recording ("clip" level).
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    recording_id: str
    recording_path: str
    sample_level: Literal["clip"] = "clip"

    t0: None = None
    t1: None = None
    label: Optional[int] = None

    meta: Dict[str, Any] = {}


@dataclass(frozen=True)
class CycleAnnotationRow:
    start: float
    end: float
    crackle: int
    wheeze: int

    @property
    def label(self) -> int:
        return int(self.crackle) + 2 * int(self.wheeze)
