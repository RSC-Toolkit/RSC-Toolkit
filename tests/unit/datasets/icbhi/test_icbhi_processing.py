import pathlib
import torch
import torchaudio

from rsc_toolkit.datasets.icbhi.processing import (
    discover_recordings,
    parse_cycle_annotations,
    make_cycle_sample_id,
    try_relpath,
)


def _write_wav(path: pathlib.Path, sr: int = 8000, seconds: float = 0.25):
    n = int(sr * seconds)
    x = torch.zeros(1, n)
    torchaudio.save(str(path), x, sampling_rate=sr)


def test_discover_recordings_pairs_only(tmp_path: pathlib.Path):
    wav1 = tmp_path / "101_1b1_Ar_sc_Meditron.wav"
    txt1 = tmp_path / "101_1b1_Ar_sc_Meditron.txt"
    _write_wav(wav1)
    txt1.write_text("0.0\t0.1\t0\t1\n")

    wav2 = tmp_path / "102_1b1_Ar_sc_Meditron.wav"
    _write_wav(wav2)
    # no txt2 on purpose

    recs = discover_recordings(tmp_path)
    assert len(recs) == 1
    assert recs[0].wav_path.name == wav1.name
    assert recs[0].txt_path.name == txt1.name


def test_parse_cycle_annotations_label(tmp_path: pathlib.Path):
    txt = tmp_path / "ann.txt"
    txt.write_text(
        "0.00\t0.50\t0\t0\n"
        "0.50\t1.00\t1\t0\n"
        "1.00\t1.50\t0\t1\n"
    )
    rows = parse_cycle_annotations(txt)
    assert len(rows) == 3
    assert rows[0].label == 0
    assert rows[1].label == 1
    assert rows[2].label == 2


def test_make_cycle_sample_id_format():
    sid = make_cycle_sample_id("rec", 0.12345, 1.5)
    assert sid == "rec_cycle_0.123_1.500"


def test_try_relpath_returns_relative_when_possible(tmp_path: pathlib.Path):
    p = tmp_path / "a.wav"
    p.write_text("x")
    assert try_relpath(p, tmp_path) == "a.wav"
