import pathlib
import json
import torch
import torchaudio

from rsc_toolkit.datasets.hflungv1.processor import HFLungV1DatasetProcessor
from rsc_toolkit.datasets.sample_index import SampleIndex


def _write_wav(path: pathlib.Path, sr: int = 8000, seconds: float = 1.0):
    n = int(sr * seconds)
    x = torch.zeros(1, n)
    torchaudio.save(str(path), x, sample_rate=sr)


def _write_annotation_csv(path: pathlib.Path, rows):
    lines = ["start,end,label"]
    for start, end, label in rows:
        lines.append(f"{start},{end},{label}")
    path.write_text("\n".join(lines) + "\n")


def test_processor_builds_indices_and_metadata(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    out_root = tmp_path / "out"
    out_root.mkdir()

    wav_train = train_dir / "rec_train_01.wav"
    wav_test = test_dir / "rec_test_01.wav"
    _write_wav(wav_train)
    _write_wav(wav_test)

    _write_annotation_csv(
        train_dir / "rec_train_01.csv",
        [(0.0, 0.5, "normal"), (0.5, 1.0, "wheeze")],
    )
    _write_annotation_csv(
        test_dir / "rec_test_01.csv",
        [(0.0, 1.0, "normal")],
    )

    metadata = data_root / "metadata.csv"
    metadata.write_text(
        "filename,split,device,location,label\n"
        "train/rec_train_01.wav,train,Litt3200,L1,normal\n"
        "test/rec_test_01.wav,test,HF_Type-1,L2,wheeze\n"
    )

    proc = HFLungV1DatasetProcessor(
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

    labels = sorted(r.label for r in cycle_idx.records)
    assert labels == [0, 0, 1]

    meta = cycle_idx.records[0].meta
    assert meta["device"] in {"Litt3200", "HF_Type-1"}
    assert meta["device_id"] in {0, 1}
    assert meta["location_id"] in {0, 1}

    splits = split_file.read_text().strip().splitlines()
    assert "rec_train_01 train" in splits
    assert "rec_test_01 test" in splits

    mapping = json.loads(label_map.read_text())
    assert mapping["normal"] == 0
    assert mapping["wheeze"] == 1


def test_processor_skips_missing_cycle_annotations(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    train_dir = data_root / "train"
    train_dir.mkdir(parents=True)
    out_root = tmp_path / "out"
    out_root.mkdir()

    wav_train = train_dir / "rec_train_01.wav"
    _write_wav(wav_train)

    metadata = data_root / "metadata.csv"
    metadata.write_text(
        "filename,split,device,location,label\n"
        "train/rec_train_01.wav,train,Litt3200,L1,normal\n"
    )

    proc = HFLungV1DatasetProcessor(
        outputs_directory=str(out_root),
        raw_data_directory=str(data_root),
    )
    proc.index_dir_path = tmp_path / "indices"

    proc.process_dataset()

    cycle_csv = proc.index_dir_path / "index_cycle.csv"
    clip_csv = proc.index_dir_path / "index_clip.csv"

    cycle_idx = SampleIndex.from_csv(cycle_csv)
    clip_idx = SampleIndex.from_csv(clip_csv)

    assert len(cycle_idx.records) == 0
    assert len(clip_idx.records) == 1
