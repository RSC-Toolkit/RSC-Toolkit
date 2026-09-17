# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
import pathlib
import pytest

from rsc_toolkit.datasets.icbhi.models import RecordingMetadata, RecordingRef


def test_recording_metadata_from_wav_path_parses_and_normalizes():
    p = pathlib.Path("101_1b1_Ar_sc_Meditron.wav")
    m = RecordingMetadata.from_wav_path(p)

    assert m.patient_id == 101
    assert m.recording_index == "1b1"
    # normalized from "Ar" -> "Anterior right" via CHEST_LOCATIONS
    assert m.chest_location == "Anterior right"
    assert m.acquisition_mode == "sc"
    assert m.device == "Meditron"
    assert m.device_id == 0
    assert m.ext == ".wav"


def test_recording_metadata_unknown_device_raises():
    p = pathlib.Path("101_1b1_Ar_sc_Unknown.wav")
    with pytest.raises(ValueError):
        RecordingMetadata.from_wav_path(p)


def test_recording_ref_from_paths_sets_recording_id(tmp_path: pathlib.Path):
    wav = tmp_path / "101_1b1_Ar_sc_Meditron.wav"
    txt = tmp_path / "101_1b1_Ar_sc_Meditron.txt"
    wav.write_bytes(b"")  # existence not required by model
    txt.write_text("0.0\t1.0\t0\t0\n")

    r = RecordingRef.from_paths(wav, txt)
    assert r.recording_id == "101_1b1_Ar_sc_Meditron"
    assert r.wav_path == wav
    assert r.txt_path == txt
    assert r.meta.patient_id == 101
