import pathlib
import json
import torch
import torchaudio

from rsc_toolkit.datasets.sprsound.processor import SPRSoundDatasetProcessor
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
