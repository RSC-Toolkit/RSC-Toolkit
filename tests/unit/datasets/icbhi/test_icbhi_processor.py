# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
import pathlib
import json
import torch
import torchaudio

from rsc_toolkit.datasets.icbhi.processor import ICBHIDatasetProcessor
from rsc_toolkit.datasets.sample_index import SampleIndex


def _write_wav(path: pathlib.Path, sr: int = 8000, seconds: float = 1.0):
    n = int(sr * seconds)
    x = torch.zeros(1, n)
    torchaudio.save(str(path), x, sample_rate=sr)


def _write_recording(tmp_path: pathlib.Path, stem: str):
    wav = tmp_path / f"{stem}.wav"
    txt = tmp_path / f"{stem}.txt"

    _write_wav(wav, sr=8000, seconds=1.0)
    txt.write_text(
        "0.00\t0.50\t0\t0\n"
        "0.50\t1.00\t1\t0\n"
    )
    return wav, txt


def test_processor_builds_cycle_and_clip_indices(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()

    _write_recording(data_root, "101_1b1_Ar_sc_Meditron")
    _write_recording(data_root, "102_1b1_Pl_sc_LittC2SE")

    proc = ICBHIDatasetProcessor(
        outputs_directory=str(out_root),
        raw_data_directory=str(data_root),
    )
    proc.index_dir_path = tmp_path / "indices"

    proc.process_dataset()

    cycle_csv = proc.index_dir_path / "index_cycle.csv"
    clip_csv = proc.index_dir_path / "index_clip.csv"

    assert cycle_csv.exists()
    assert clip_csv.exists()

    cycle_idx = SampleIndex.from_csv(cycle_csv)
    clip_idx = SampleIndex.from_csv(clip_csv)

    # 2 recordings * 2 cycles each
    assert len(cycle_idx.records) == 4
    assert len(clip_idx.records) == 2

    # labels: [0, 1] repeated
    labels = sorted(r.label for r in cycle_idx.records)
    assert labels == [0, 0, 1, 1]


def test_processor_writes_stats_files(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()

    _write_recording(data_root, "101_1b1_Ar_sc_Meditron")

    proc = ICBHIDatasetProcessor(outputs_directory=str(out_root), raw_data_directory=str(data_root))
    proc.index_dir_path = tmp_path / "indices"

    proc.process_dataset()

    stats_cycle = proc.index_dir_path / "index_cycle_stats.json"
    stats_clip = proc.index_dir_path / "index_clip_stats.json"

    assert stats_cycle.exists()
    assert stats_clip.exists()

    cycle_stats = json.loads(stats_cycle.read_text())
    clip_stats = json.loads(stats_clip.read_text())

    assert cycle_stats["total_samples"] == 2
    assert cycle_stats["sample_levels"]["cycle"] == 2
    assert cycle_stats["label_counts"]["0"] == 1
    assert cycle_stats["label_counts"]["1"] == 1

    assert clip_stats["total_samples"] == 1
    assert clip_stats["sample_levels"]["clip"] == 1


def test_processor_attaches_metadata_if_present(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()

    _write_recording(data_root, "101_1b1_Ar_sc_Meditron")

    # metadata.txt
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    metadata_file = meta_dir / "metadata.txt"
    metadata_file.write_text(
        "101\t55\tM\t24.5\t\t\tAr\n"
    )

    proc = ICBHIDatasetProcessor(
        outputs_directory=str(out_root),
        raw_data_directory=str(data_root),
        metadata_file=str(metadata_file),
    )
    proc.index_dir_path = tmp_path / "indices"

    proc.process_dataset()

    cycle_idx = SampleIndex.from_csv(proc.index_dir_path / "index_cycle.csv")
    rec = cycle_idx.records[0]

    assert rec.meta["patient_id"] == 101
    assert rec.meta["age"] == 55
    assert rec.meta["sex"] == 0
    assert rec.meta["adult_BMI"] == 24.5
    assert rec.meta["chest_location_id"] == 2


def test_processor_handles_missing_metadata_gracefully(tmp_path: pathlib.Path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()

    _write_recording(data_root, "101_1b1_Ar_sc_Meditron")

    proc = ICBHIDatasetProcessor(
        outputs_directory=str(out_root),
        raw_data_directory=str(data_root),
        metadata_file=str(tmp_path / "definitely_not_there" / "metadata.txt"),
        )
    proc.index_dir_path = tmp_path / "indices"

    # No metadata file provided
    proc.process_dataset()

    cycle_idx = SampleIndex.from_csv(proc.index_dir_path / "index_cycle.csv")
    rec = cycle_idx.records[0]

    # still has base meta
    assert rec.meta["patient_id"] == 101
    assert "age" in rec.meta
    assert rec.meta["age"] is None
