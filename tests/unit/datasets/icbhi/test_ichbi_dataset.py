import pathlib
import pytest
import torch
import torchaudio

from rsc_toolkit.datasets.sample_index import SampleIndex, SampleRecord
from rsc_toolkit.datasets.index_dataset import IndexDataset


def _write_wav(path: pathlib.Path, sr: int = 8000, seconds: float = 1.0):
    n = int(sr * seconds)
    x = torch.arange(n, dtype=torch.float32).unsqueeze(0)  # [1, T]
    torchaudio.save(str(path), x, sample_rate=sr)


def test_cycle_item_loads_slices_and_decodes_meta(tmp_path: pathlib.Path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    wav = data_root / "101_1b1_Ar_sc_Meditron.wav"
    _write_wav(wav, sr=8000, seconds=1.0)

    idx = SampleIndex(
        records=[
            SampleRecord(
                sample_id="101_1b1_Ar_sc_Meditron_cycle_0.250_0.750",
                recording_path=wav.name,  # relative to data_root
                sample_level="cycle",
                t0=0.25,
                t1=0.75,
                label=2,
                meta={"patient_id": 101, "device": "Meditron"},
            )
        ],
        data_root=str(data_root),
    )
    index_csv = tmp_path / "index_cycle.csv"
    idx.to_csv(index_csv)

    ds = IndexDataset(
        index_csv=index_csv,
        data_root=data_root,
        sampling_rate=8000,
        sample_level="cycle",
        return_meta=True,
        fade=False,  # deterministic length
    )

    item = ds[0]
    assert item["sample_id"].endswith("0.250_0.750")
    assert item["sample_level"] == "cycle"
    assert item["label"] == 2
    assert item["meta"]["patient_id"] == 101

    audio = item["audio"]
    assert isinstance(audio, torch.Tensor)
    assert audio.shape[0] == 1
    # 0.5 seconds * 8000 Hz = 4000 samples
    assert audio.shape[1] == 4000


def test_clip_item_loads_full_recording(tmp_path: pathlib.Path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    wav = data_root / "101_1b1_Ar_sc_Meditron.wav"
    _write_wav(wav, sr=8000, seconds=1.0)

    idx = SampleIndex(
        records=[
            SampleRecord(
                sample_id=wav.stem,
                recording_path=wav.name,
                sample_level="clip",
                t0=None,
                t1=None,
                label=None,
                meta={"patient_id": 101},
            )
        ],
        data_root=str(data_root),
    )
    index_csv = tmp_path / "index_clip.csv"
    idx.to_csv(index_csv)

    ds = IndexDataset(
        index_csv=index_csv,
        data_root=data_root,
        sampling_rate=8000,
        sample_level="clip",
        return_meta=True,
        fade=False,
    )

    item = ds[0]
    assert item["sample_level"] == "clip"
    assert item["label"] is None
    assert item["audio"].shape == (1, 8000)
    assert item["meta"]["patient_id"] == 101


def test_resolve_recording_path_defaults_to_index_parent_when_data_root_missing(tmp_path: pathlib.Path):
    # No data_root passed -> uses index_csv.parent / recording_path
    wav = tmp_path / "101_1b1_Ar_sc_Meditron.wav"
    _write_wav(wav, sr=8000, seconds=0.25)

    idx = SampleIndex(
        records=[
            SampleRecord(
                sample_id=wav.stem,
                recording_path=wav.name,  # relative to index dir
                sample_level="clip",
                meta={},
            )
        ]
    )
    index_csv = tmp_path / "index_clip.csv"
    idx.to_csv(index_csv)

    ds = IndexDataset(index_csv=index_csv, data_root=None, sampling_rate=8000, fade=False)
    item = ds[0]
    assert item["audio"].shape == (1, 2000)  # 0.25 sec * 8000


def test_return_meta_false_omits_meta_key(tmp_path: pathlib.Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    wav = data_root / "101_1b1_Ar_sc_Meditron.wav"
    _write_wav(wav, sr=8000, seconds=0.25)

    idx = SampleIndex(
        records=[SampleRecord(sample_id=wav.stem, recording_path=wav.name, sample_level="clip", meta={"a": 1})],
        data_root=str(data_root),
    )
    index_csv = tmp_path / "index_clip.csv"
    idx.to_csv(index_csv)

    ds = IndexDataset(index_csv=index_csv, data_root=data_root, sampling_rate=8000, return_meta=False, fade=False)
    item = ds[0]
    assert "meta" not in item


def test_invalid_sample_level_init_raises(tmp_path: pathlib.Path):
    # minimal valid csv with required columns but invalid sample_level argument
    index_csv = tmp_path / "index.csv"
    index_csv.write_text(
        "sample_id,recording_path,sample_level,t0,t1,label,meta\n"
        "a,a.wav,cycle,0.0,1.0,0,{}\n"
    )

    with pytest.raises(ValueError, match="Invalid sample_level"):
        IndexDataset(index_csv=index_csv, sample_level="bad")  # type: ignore[arg-type]


def test_missing_required_columns_raises(tmp_path: pathlib.Path):
    index_csv = tmp_path / "bad.csv"
    index_csv.write_text("sample_id,recording_path\nx,y\n")

    with pytest.raises(ValueError, match="missing columns"):
        IndexDataset(index_csv=index_csv)


def test_split_filters_rows_with_custom_split_file(tmp_path: pathlib.Path):
    index_csv = tmp_path / "index_cycle.csv"
    idx = SampleIndex(
        records=[
            SampleRecord(
                sample_id="rec_a_cycle_0.000_1.000",
                recording_path="rec_a.wav",
                sample_level="cycle",
                t0=0.0,
                t1=1.0,
                label=0,
                meta={},
            ),
            SampleRecord(
                sample_id="rec_b_cycle_0.000_1.000",
                recording_path="rec_b.wav",
                sample_level="cycle",
                t0=0.0,
                t1=1.0,
                label=1,
                meta={},
            ),
        ]
    )
    idx.to_csv(index_csv)

    split_file = tmp_path / "split.txt"
    split_file.write_text("rec_a train\nrec_b test\n")

    ds_train = IndexDataset(index_csv=index_csv, split="train", split_file=split_file)
    assert len(ds_train) == 1
    assert str(ds_train.df.iloc[0]["sample_id"]).startswith("rec_a")

    ds_test = IndexDataset(index_csv=index_csv, split="test", split_file=split_file)
    assert len(ds_test) == 1
    assert str(ds_test.df.iloc[0]["sample_id"]).startswith("rec_b")

@pytest.mark.skip(reason="data not downloaded by default.")
def test_split_filters_rows_using_packaged_official_split_file(tmp_path: pathlib.Path):
    """
    Smoke test: packaged `official_split.txt` is used by default.
    This test does not touch audio; it only checks row filtering by `recording_path` stem.
    """
    recording_id = "101_1b1_Al_sc_Meditron"

    index_csv = tmp_path / "index_clip.csv"
    idx = SampleIndex(
        records=[
            SampleRecord(
                sample_id=recording_id,
                recording_path=f"{recording_id}.wav",
                sample_level="clip",
                t0=None,
                t1=None,
                label=None,
                meta={},
            )
        ]
    )
    idx.to_csv(index_csv)

    ds_test = IndexDataset(index_csv=index_csv, split="test")
    assert len(ds_test) == 1

    ds_train = IndexDataset(index_csv=index_csv, split="train")
    assert len(ds_train) == 0
