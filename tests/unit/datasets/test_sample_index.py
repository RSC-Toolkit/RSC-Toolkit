# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
import json
import pathlib

from rsc_toolkit.datasets.sample_index import SampleIndex, SampleRecord


def test_sample_index_to_csv_and_from_csv_roundtrip(tmp_path: pathlib.Path):
    records = [
        SampleRecord(
            sample_id="r1_cycle_0.000_1.000",
            recording_path="r1.wav",
            sample_level="cycle",
            t0=0.0,
            t1=1.0,
            label=2,
            meta={"patient_id": 101, "device": "Meditron"},
        ),
        SampleRecord(
            sample_id="r2",
            recording_path="r2.wav",
            sample_level="clip",
            t0=None,
            t1=None,
            label=None,
            meta={"patient_id": 102},
        ),
    ]
    idx = SampleIndex(records=records, data_root=str(tmp_path / "data"))

    out_csv = tmp_path / "index.csv"
    idx.to_csv(out_csv)

    loaded = SampleIndex.from_csv(out_csv)
    assert len(loaded.records) == 2

    r0 = loaded.records[0]
    assert r0.sample_id == "r1_cycle_0.000_1.000"
    assert r0.sample_level == "cycle"
    assert r0.t0 == 0.0
    assert r0.t1 == 1.0
    assert r0.label == 2
    assert r0.meta["patient_id"] == 101
    assert r0.meta["device"] == "Meditron"

    r1 = loaded.records[1]
    assert r1.sample_level == "clip"
    assert r1.t0 is None
    assert r1.label is None
    assert r1.meta["patient_id"] == 102


def test_sample_index_stats(tmp_path: pathlib.Path):
    idx = SampleIndex(
        records=[
            SampleRecord(sample_id="a", recording_path="a.wav", sample_level="clip", meta={}),
            SampleRecord(sample_id="b", recording_path="b.wav", sample_level="cycle", label=0, t0=0.0, t1=1.0, meta={}),
            SampleRecord(sample_id="c", recording_path="c.wav", sample_level="cycle", label=2, t0=0.0, t1=1.0, meta={}),
            SampleRecord(sample_id="d", recording_path="d.wav", sample_level="cycle", label=2, t0=0.0, t1=1.0, meta={}),
        ]
    )
    s = idx.stats()
    assert s["total_samples"] == 4
    assert s["sample_levels"]["cycle"] == 3
    assert s["sample_levels"]["clip"] == 1
    assert s["label_counts"]["2"] == 2
    assert s["label_counts"]["0"] == 1


def test_sample_index_write_stats(tmp_path: pathlib.Path):
    idx = SampleIndex(records=[])
    out = tmp_path / "stats.json"
    idx.write_stats(out)

    data = json.loads(out.read_text())
    assert data["total_samples"] == 0
    assert data["sample_levels"] == {}
    assert data["label_counts"] == {}
