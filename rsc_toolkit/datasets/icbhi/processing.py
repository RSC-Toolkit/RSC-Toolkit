# rsc_toolkit/datasets/icbhi/processing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Literal
import pathlib
import logging

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

from .models import RecordingRef, CycleAnnotationRow

logger = logging.getLogger(__name__)


# ----------------------------
# Discovery / metadata attach
# ----------------------------

def discover_recordings(data_folder: pathlib.Path) -> List[RecordingRef]:
    """
    Discover paired .wav + .txt files and return RecordingRef objects.
    """
    data_folder = data_folder.expanduser().resolve()
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    wavs = sorted(data_folder.glob("*.wav"))
    if not wavs:
        logger.warning("No .wav files found in %s", data_folder)

    recordings: List[RecordingRef] = []
    for wav_path in wavs:
        txt_path = data_folder / f"{wav_path.stem}.txt"
        if not txt_path.exists():
            logger.warning("Missing annotation .txt for %s", wav_path.name)
            continue
        try:
            recordings.append(RecordingRef.from_paths(wav_path, txt_path))
        except Exception as e:
            logger.warning("Skipping %s due to filename parse error: %s", wav_path.name, e)

    return recordings


def load_patient_metadata(metadata_path: pathlib.Path) -> pd.DataFrame:
    """
    Load ICBHI metadata.txt as a DataFrame indexed by patient_id.
    """
    metadata_path = metadata_path.expanduser().resolve()
    df = pd.read_csv(
        metadata_path,
        names=["age", "sex", "adult_BMI", "child_weight", "child_height", "chest_location"],
        delimiter="\t",
        index_col=0,  # patient_id
    )

    # Avoid pandas FutureWarning downcasting changes
    pd.set_option("future.no_silent_downcasting", True)

    df["sex"] = df["sex"].replace({"M": 0, "F": 1}).infer_objects(copy=False)
    df["chest_location"] = df["chest_location"].replace(
        {"Tc": 0, "Al": 1, "Ar": 2, "Pl": 3, "Pr": 4, "Ll": 5, "Lr": 6}
    ).infer_objects(copy=False)

    return df


def attach_patient_metadata(recordings: Iterable[RecordingRef], df: pd.DataFrame) -> None:
    """
    Mutate RecordingRef.meta in-place by attaching fields from metadata df.

    This is still "processing": we keep it out of models.py.
    """
    for rec in recordings:
        pid = rec.meta.patient_id
        if pid not in df.index:
            logger.warning("No metadata found for patient_id=%s", pid)
            continue

        row = df.loc[pid]
        rec.meta.age = row["age"]
        rec.meta.sex = row["sex"]
        rec.meta.adult_BMI = row["adult_BMI"]
        rec.meta.child_weight = row["child_weight"]
        rec.meta.child_height = row["child_height"]
        rec.meta.chest_location_id = row["chest_location"]


# ----------------------------
# Annotation parsing
# ----------------------------

def parse_cycle_annotations(txt_path: pathlib.Path) -> List[CycleAnnotationRow]:
    """
    Parse an ICBHI per-recording annotation file.

    Expected columns (tab-separated, no header):
        Start   End     Crackle Wheeze
    """
    df = pd.read_csv(txt_path, delimiter="\t", header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Annotation file has <4 columns: {txt_path}")

    df = df.iloc[:, :4]
    df.columns = ["Start", "End", "Crackle", "Wheeze"]

    rows: List[CycleAnnotationRow] = []
    for _, r in df.iterrows():
        rows.append(
            CycleAnnotationRow(
                start=float(r["Start"]),
                end=float(r["End"]),
                crackle=int(r["Crackle"]),
                wheeze=int(r["Wheeze"]),
            )
        )
    return rows


def make_cycle_sample_id(recording_id: str, t0: float, t1: float) -> str:
    return f"{recording_id}_cycle_{t0:.3f}_{t1:.3f}"


def build_meta_dict(rec: RecordingRef) -> Dict[str, Any]:
    """
    Denormalize metadata into a JSON-friendly dict for indexing.
    """
    m = rec.meta
    return {
        "patient_id": m.patient_id,
        "recording_index": m.recording_index,
        "chest_location": m.chest_location,
        "acquisition_mode": m.acquisition_mode,
        "device": m.device,
        "device_id": m.device_id,
        "chest_location_id": m.chest_location_id,
        "age": m.age,
        "sex": m.sex,
        "adult_BMI": m.adult_BMI,
        "child_weight": m.child_weight,
        "child_height": m.child_height,
    }


def try_relpath(path: pathlib.Path, root: pathlib.Path) -> str:
    """
    Store a relative path if possible, otherwise absolute.
    """
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ----------------------------
# Audio ops (kept out of models)
# ----------------------------

def load_audio(
    wav_path: pathlib.Path,
    target_sampling_rate: int,
    fade: bool = True,
    fade_fraction: float = 1 / 16,
) -> torch.Tensor:
    """
    Load waveform as torch.Tensor [channels, samples], resample if needed,
    and optionally apply a small fade in/out to reduce boundary clicks.
    """
    data, sr = torchaudio.load(str(wav_path))
    if sr != target_sampling_rate:
        data = T.Resample(sr, target_sampling_rate)(data)

    if fade:
        fade_samples = int(target_sampling_rate * fade_fraction)
        if fade_samples > 0:
            data = T.Fade(
                fade_in_len=fade_samples,
                fade_out_len=fade_samples,
                fade_shape="linear",
            )(data)

    return data


def slice_audio(
    data: torch.Tensor,
    t0: float,
    t1: float,
    sampling_rate: int,
) -> torch.Tensor:
    """
    Slice waveform by time. Clamps to valid range.
    """
    if t1 < t0:
        raise ValueError(f"Invalid segment: t1({t1}) < t0({t0})")

    max_ind = int(data.shape[1])
    start_ind = max(0, min(int(t0 * sampling_rate), max_ind))
    end_ind = max(0, min(int(t1 * sampling_rate), max_ind))

    if end_ind < start_ind:
        end_ind = start_ind

    return data[:, start_ind:end_ind]


# ----------------------------
# Optional: batch cycle extraction
# ----------------------------

@dataclass(frozen=True)
class CycleAudio:
    """
    Runtime object: audio + segment info.
    Kept in processing.py (not models.py) because it holds torch.Tensor.
    """
    sample_id: str
    recording_id: str
    t0: float
    t1: float
    label: int
    audio: torch.Tensor


def extract_cycles_audio_for_recording(
    rec: RecordingRef,
    data_root: pathlib.Path,
    sampling_rate: int,
) -> List[CycleAudio]:
    """
    Convenience helper for eager extraction (debugging/feature preprocessing).
    For scalable training, you typically *don't* do this in the processor;
    you load slices in the Dataset __getitem__.
    """
    ann = parse_cycle_annotations(rec.txt_path)
    wav = load_audio(rec.wav_path, target_sampling_rate=sampling_rate)

    out: List[CycleAudio] = []
    for row in ann:
        sid = make_cycle_sample_id(rec.recording_id, row.start, row.end)
        audio = slice_audio(wav, row.start, row.end, sampling_rate)
        out.append(
            CycleAudio(
                sample_id=sid,
                recording_id=rec.recording_id,
                t0=row.start,
                t1=row.end,
                label=row.label,
                audio=audio,
            )
        )
    return out
