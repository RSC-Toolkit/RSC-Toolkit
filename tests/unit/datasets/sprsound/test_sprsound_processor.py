import pathlib
import json
import torch
import torchaudio

import pytest

from rsc_toolkit.datasets.sprsound.processor import SPRSoundDatasetProcessor
from rsc_toolkit.datasets.sprsound.processing import normalize_recording_id
from rsc_toolkit.datasets.sample_index import SampleIndex


def _write_wav(path: pathlib.Path, sr: int = 8000, seconds: float = 1.0):
    n = int(sr * seconds)
    x = torch.zeros(1, n)
    torchaudio.save(str(path), x, sample_rate=sr)


def test_sprsound_processor_builds_indices(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    class_root = data_root / "Classification"
    train_dir = class_root / "train_classification_wav"
    valid_dir = class_root / "valid_classification_wav" / "2022"
    train_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)

    out_root = tmp_path / "out"
    out_root.mkdir()

    _write_wav(train_dir / "train_01.wav")
    _write_wav(valid_dir / "valid_01.wav")

    metadata = class_root / "metadata.csv"
    metadata.write_text(
        "Filename,Patient_number,Age,Gender,Recording_location,Recording_number,Record_annotation,Start,End,Label,Split\n"
        "train_01,1001,10,0,p1,1,Normal,0.0,0.5,Normal,train\n"
        "train_01,1001,10,0,p1,1,Normal,0.5,1.0,Wheeze,train\n"
        "valid_01,1002,12,1,p2,2,CAS,0.0,1.0,Wheeze,valid\n"
    )

    proc = SPRSoundDatasetProcessor(
        outputs_directory=str(out_root),
        raw_data_directory=str(data_root),
    )
    proc.index_dir_path = tmp_path / "indices"

    proc.process_dataset()

    cycle_csv = proc.index_dir_path / "index_cycle.csv"
    clip_csv = proc.index_dir_path / "index_clip.csv"
    split_file = proc.index_dir_path / "official_split.txt"
    label_map = proc.index_dir_path / "label_map.json"

    assert cycle_csv.exists()
    assert clip_csv.exists()
    assert split_file.exists()
    assert label_map.exists()

    cycle_idx = SampleIndex.from_csv(cycle_csv)
    clip_idx = SampleIndex.from_csv(clip_csv)

    assert len(cycle_idx.records) == 3
    assert len(clip_idx.records) == 2

    splits = {r.split for r in cycle_idx.records}
    assert splits == {"train", "test"}

    meta = cycle_idx.records[0].meta
    assert meta["device"] == "Yunting2"
    assert meta["device_id"] == 0
    assert meta["location_id"] in {0, 1}

    mapping = json.loads(label_map.read_text())
    assert "Normal" in mapping
    assert "Wheeze" in mapping
    assert "CAS" in mapping


@pytest.mark.parametrize(
    "value, expected",
    [
        # SPRSound 실제 파일명: <환자번호>_<나이>_<성별>_<위치>_<녹음번호>.
        # 나이가 소수라 이름 중간에 점이 있다. 이걸 확장자로 오인해 자르면 안 된다.
        ("40138127_14.7_0_p3_139", "40138127_14.7_0_p3_139"),
        ("40138127_14.7_0_p3_139.wav", "40138127_14.7_0_p3_139"),
        ("40138127_14.7_0_p3_139.json", "40138127_14.7_0_p3_139"),
        ("a/b/40138127_14.7_0_p3_139.wav", "40138127_14.7_0_p3_139"),
        ("train_01.wav", "train_01"),
        ("train_01", "train_01"),
    ],
)
def test_normalize_recording_id_keeps_dots_inside_name(value, expected):
    assert normalize_recording_id(value) == expected


def test_processor_resolves_filenames_with_decimal_age(tmp_path: pathlib.Path):
    """
    나이에 소수점이 든 실제 형식의 파일명으로 인덱스가 채워지는지 확인한다.

    이 결함이 있을 때는 metadata 전 행이 'Skipping missing audio file' 로 빠져
    인덱스가 헤더만 남은 채 build-index 가 성공(exit 0)했다.
    """
    data_root = tmp_path / "raw"
    class_root = data_root / "Classification"
    train_dir = class_root / "train_classification_wav"
    valid_dir = class_root / "valid_classification_wav" / "2022"
    train_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)

    out_root = tmp_path / "out"
    out_root.mkdir()

    train_id = "40138127_14.7_0_p3_139"
    valid_id = "65121853_1.5_0_p4_4246"
    _write_wav(train_dir / f"{train_id}.wav")
    _write_wav(valid_dir / f"{valid_id}.wav")

    metadata = class_root / "metadata.csv"
    metadata.write_text(
        "Filename,Patient_number,Age,Gender,Recording_location,Recording_number,"
        "Record_annotation,Start,End,Label,Split\n"
        f"{train_id},40138127,14.7,0,p3,139,Normal,0.0,0.5,Normal,train\n"
        f"{valid_id},65121853,1.5,0,p4,4246,CAS,0.0,1.0,Wheeze,valid\n"
    )

    proc = SPRSoundDatasetProcessor(
        outputs_directory=str(out_root),
        raw_data_directory=str(data_root),
    )
    proc.index_dir_path = tmp_path / "indices"
    proc.process_dataset()

    cycle_idx = SampleIndex.from_csv(proc.index_dir_path / "index_cycle.csv")
    clip_idx = SampleIndex.from_csv(proc.index_dir_path / "index_clip.csv")

    assert len(cycle_idx.records) == 2
    assert len(clip_idx.records) == 2
    assert {r.sample_id for r in clip_idx.records} == {train_id, valid_id}
    assert {r.split for r in clip_idx.records} == {"train", "test"}
