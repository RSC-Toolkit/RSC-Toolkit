"""
Official-code parity golden tests for the SCL-metadata group (CE, SCL, Hybrid, M-SCL; CNN6).

Golden source: scripts/reproduction/dump_sclmeta_goldens.py — the official ilyassmoummad/scl_icbhi2017 code run in our venv.
"""
import json
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.augmentations.specaugment_torchaudio import SpecAugmentTA
from rsc_toolkit.methods.sgscl import MetaCL as SupConLoss
from rsc_toolkit.processors.sclmeta_mel import SCLMetaMelProcessor, batch_minmax, standardize


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "sclmeta"
DATA = REPO / ".datasets" / "icbhi"
PANN = REPO / "pretrained_models" / "upstream" / "Cnn6_mAP=0.343.pth"
needs_golden = pytest.mark.skipif(not (GOLDEN / "features.pt").exists(), reason="golden missing")
needs_data = pytest.mark.skipif(not (DATA / "index" / "index_cycle.csv").exists(), reason="ICBHI index missing")
needs_pann = pytest.mark.skipif(not PANN.exists(), reason="PANN Cnn6 ckpt missing")


@needs_golden
@needs_data
def test_audio_pipeline_matches_reference():
    """Segment load -> resample -> fade in/out -> circular padding -> fade_out (dataset.py:93-124) is bit-identical to the IndexDataset path."""
    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    golden = torch.load(GOLDEN / "audio.pt")
    pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=8.0, pad_type="sclmeta"))
    ds = IndexDataset(index_csv=DATA / "index/index_cycle.csv", data_root=DATA, sampling_rate=16000,
                      sample_level="cycle", audio_preprocessor=pre, fade=False, resample_after_slice=True)
    key = {(str(r.recording_path), round(float(r.t0), 3), round(float(r.t1), 3)): i for i, r in enumerate(ds.df.itertuples())}
    for g in golden:
        i = key[(g["filepath"], round(g["onset"], 3), round(g["offset"], 3))]
        item = ds[i]
        assert item["label"] == g["label"]
        assert item["audio"].shape == (1, 128000)
        assert torch.equal(item["audio"][0], g["wav"]), g["filepath"]


@needs_golden
@pytest.mark.exact_golden  # bitwise comparison — dev machine only (CPUs differ across CI runners, so the last digits diverge)
def test_feature_stages_match_reference():
    g = torch.load(GOLDEN / "features.pt")
    proc = SCLMetaMelProcessor()
    dbmel = torch.stack([proc.make_features(w) for w in g["wav"]])  # (N,1,64,251)
    assert torch.equal(dbmel, g["dbmel"])
    val = standardize(batch_minmax(dbmel))
    assert torch.equal(val, g["val"])
    sa = SpecAugmentTA(20, 40, 2, 2, 1.0)
    torch.manual_seed(g["seed"])
    v1 = standardize(sa(batch_minmax(dbmel)))
    v2 = standardize(sa(batch_minmax(dbmel)))
    assert torch.equal(v1, g["view1"]) and torch.equal(v2, g["view2"])


@needs_golden
@needs_pann
def test_cnn6_embedding_matches_reference():
    from rsc_toolkit.models.cnn6_sclmeta import CNN6SCLMeta

    g = torch.load(GOLDEN / "features.pt"); c = torch.load(GOLDEN / "cnn6.pt")
    m = CNN6SCLMeta(method="scl").eval()
    with torch.no_grad():
        emb = m.forward_features(g["val"])
    assert torch.allclose(emb, c["emb"], atol=1e-6), float((emb - c["emb"]).abs().max())
    ours = {k: list(v.shape) for k, v in m.encoder.state_dict().items()}
    # original embed_only model keys have a prefix (e.g. cnn6.) followed by conv_block*, plus 2 unused linear.* keys (26 keys)
    ref = {k[k.index("conv_block"):]: v for k, v in c["state_dict_shapes"].items() if "conv_block" in k}
    assert ours == ref and len(c["state_dict_shapes"]) == 26


@needs_golden
def test_losses_match_reference():
    from rsc_toolkit.trainer.losses import WeightedCrossEntropyFromCounts

    g = torch.load(GOLDEN / "losses.pt")
    sup = SupConLoss(temperature=0.06)
    assert torch.equal(sup(g["p1"], g["p2"], g["y"]), g["supcon_labels"])
    assert torch.equal(sup(g["p1"], g["p2"], None), g["supcon_none"])
    ce = WeightedCrossEntropyFromCounts([2063, 1215, 501, 363])
    assert torch.allclose(ce.weight, g["weights"])
    hybrid = 0.5 * sup(g["p1"], g["p2"], g["y"]) + 0.5 * ce(torch.cat([g["pred1"], g["pred2"]]), torch.cat([g["y"], g["y"]]))
    assert torch.allclose(hybrid, g["supconce"], atol=1e-6)
    mscl = 0.25 * sup(g["p1"], g["p2"], g["meta"]) + 0.75 * sup(g["pred1"], g["pred2"], g["y"])
    assert torch.allclose(mscl, g["mscl"], atol=1e-6)


@needs_golden
@needs_data
def test_meta_sa_labels_match_reference_csv_for_all_cycles():
    import pandas as pd
    from rsc_toolkit.datasets.meta_labels import build_meta_label_fn

    ref = json.load(open(GOLDEN / "meta_sa.json"))
    df = pd.read_csv(DATA / "index/index_cycle.csv")
    metas = df["meta"].map(json.loads)
    ours = {(str(r.recording_path), round(float(r.t0), 3), round(float(r.t1), 3)): build_meta_label_fn("sa")({"meta": m})
            for r, m in zip(df.itertuples(), metas)}
    mism = [(r["filepath"], r["onset"]) for r in ref if ours[(r["filepath"], round(r["onset"], 3), round(r["offset"], 3))] != r["sa"]]
    assert len(ref) == 6898 and not mism, mism[:5]


def test_methods_forward_shapes_and_linear_freeze(tmp_path):
    from rsc_toolkit.models.cnn6_sclmeta import CNN6SCLMeta

    x = torch.randn(6, 64, 251) * 10 - 40
    y = torch.tensor([0, 1, 2, 3, 0, 1]); meta = torch.tensor([0, 0, 1, 2, 3, 3])
    ce = nn.CrossEntropyLoss()
    for method in ("ce", "hybrid", "scl", "mscl"):
        m = CNN6SCLMeta(method=method, from_sl_official=False).train()
        logits, loss = m.training_forward(x, y, ce, meta_labels=meta)
        assert logits.shape == (6, 4) and torch.isfinite(loss) and loss.requires_grad, method
    pre = CNN6SCLMeta(method="mscl", from_sl_official=False)
    torch.save({"model": pre.state_dict()}, tmp_path / "final_state.pth")
    lin = CNN6SCLMeta(method="linear", encoder_ckpt=str(tmp_path / "final_state.pth")).train()
    assert all(not p.requires_grad for p in lin.encoder.parameters()) and not lin.encoder.training
    logits, loss = lin.training_forward(x, y, ce)
    loss.backward()
    assert lin.classifier.weight.grad is not None
    assert torch.equal(lin.encoder.conv_block1.conv1.weight, pre.encoder.conv_block1.conv1.weight)
    with pytest.raises(ValueError, match="encoder_ckpt"):
        CNN6SCLMeta(method="linear")


def test_specaugment_gate_uses_randn_quirk():
    sa = SpecAugmentTA(p=1.0)
    x = torch.rand(64, 1, 64, 251)
    torch.manual_seed(0)
    applied = sum(int(not torch.equal(sa(x), x)) for _ in range(200))
    assert 140 < applied < 195  # p=1.0, but the randn gate gives ≈84%


@needs_data
def test_segment_loading_equals_full_load_then_slice():
    """frame_offset/num_frames segment loading yields the same samples as a full load followed by slicing (including boundaries)."""
    import torchaudio
    from rsc_toolkit.datasets.icbhi.processing import slice_audio
    from rsc_toolkit.datasets.index_dataset import IndexDataset

    ds = IndexDataset(index_csv=DATA / "index/index_cycle.csv", data_root=DATA, sampling_rate=16000,
                      sample_level="cycle", fade=False, resample_after_slice=True)
    for i in (0, 1, 2500, len(ds) - 1):
        row = ds.df.iloc[i]
        wav, sr = torchaudio.load(str(DATA / row["recording_path"]))
        ref = slice_audio(wav, t0=float(row["t0"]), t1=float(row["t1"]), sampling_rate=sr)
        if sr != 16000:
            ref = torchaudio.transforms.Resample(sr, 16000)(ref)
        assert torch.equal(ds[i]["audio"], ref), row["sample_id"]
