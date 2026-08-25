"""
SPRSound(2022 inter-patient) 을 SCL-metadata 경로로 읽는 P0 테스트 — 공식 dataset.py SPRS 와 대조.
골든: scripts/reproduction/dump_sclmeta_sprs_goldens.py (audio.pt 8 이벤트 파형, rows.json 8085 행 라벨·meta).
"""
import json
import pathlib

import pandas as pd
import pytest
import torch

from rsc_toolkit.datasets.meta_labels import build_meta_label_fn


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "sclmeta_sprs"
DATA = REPO / ".datasets" / "sprsound"
needs = pytest.mark.skipif(not ((GOLDEN / "audio.pt").exists() and (DATA / "index" / "index_cycle.csv").exists()), reason="골든/데이터 없음")

SPRS_CLASS_DICT = {"Normal": 0, "Fine Crackle": 1, "Wheeze": 2, "Coarse Crackle": 3, "Wheeze+Crackle": 4, "Rhonchi": 5, "Stridor": 6}


def _index_key(rec_path, t0, t1):
    return (pathlib.Path(rec_path).name, int(round(float(t0) * 1000)), int(round(float(t1) * 1000)))


@needs
def test_event_waveforms_match_reference():
    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    golden = torch.load(GOLDEN / "audio.pt")
    pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=8.0, pad_type="sclmeta"))
    ds = IndexDataset(index_csv=DATA / "index/index_cycle.csv", data_root=DATA / "Classification", sampling_rate=16000, sample_level="cycle",
                      audio_preprocessor=pre, fade=False, resample_after_slice=True, time_unit="ms_int")
    key = {_index_key(r.recording_path, r.t0, r.t1): i for i, r in enumerate(ds.df.itertuples())}
    for g in golden:
        i = key[(g["wav_path"], g["onset"], g["offset"])]
        item = ds[i]
        assert item["audio"].shape == (1, 128000)
        assert torch.equal(item["audio"][0], g["wav"]), (g["wav_path"], g["onset"])
        assert item["label"] == g["label"]
        assert build_meta_label_fn("sa_sprs")(item) == g["meta"]


@needs
def test_all_rows_labels_split_and_meta_match_reference_csv():
    rows = json.load(open(GOLDEN / "rows.json"))
    df = pd.read_csv(DATA / "index/index_cycle.csv")
    fn = build_meta_label_fn("sa_sprs")
    ours = {_index_key(r.recording_path, r.t0, r.t1): (int(r.label), str(r.split), fn({"meta": json.loads(r.meta)}))
            for r in df.itertuples()}
    ref = [r for r in rows if r["split"] in ("train", "inter_test")]
    assert len(ref) == 8085 == len(ours)
    bad = []
    for r in ref:
        k = (r["wav_path"], r["onset"], r["offset"])
        want = (r["label"], "train" if r["split"] == "train" else "test", r["sa"])
        if ours.get(k) != want:
            bad.append((k, ours.get(k), want))
    assert not bad, bad[:5]
