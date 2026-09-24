"""
Official-code parity golden and configuration tests for OPERA-CT (HTSAT-Swin) + the EMBC'25 frozen evaluation path.

Golden source: scripts/reproduction/dump_opera_goldens.py — the official evelyn0414/OPERA Cola(htsat) and eval-audio-repr evar/ar_opera, solve.py
run in our venv. The backbone is a toolkit re-implementation (models/opera/htsat.py); the official ckpt is loaded strictly and embeddings are compared.
"""
import pathlib

import pytest
import torch

from rsc_toolkit.models.opera import HTSATSwin, load_opera_ct_checkpoint
from rsc_toolkit.models.opera_ct import MLPHead, OperaCT, TfmEncHead
from rsc_toolkit.processors.opera_mel import OperaBatchMel, librosa_trim_indices


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "opera"
CKPT = REPO / "pretrained_models/upstream/opera/encoder-operaCT.ckpt"
needs_golden = pytest.mark.skipif(not (GOLDEN / "features.pt").exists(), reason="golden missing")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="OPERA-CT ckpt missing")


def test_htsat_layout():
    m = HTSATSwin()
    sd = m.state_dict()
    assert len(sd) == 197 and m.num_features == 768 and m.freq_ratio == 4
    assert m.patch_embed.grid_size == (64, 64)
    # last stage (8×8): window 8 = resolution -> shift 0 (original quirk)
    assert all(b.shift_size == 0 for b in m.layers[3].blocks)
    assert m.layers[0].blocks[1].shift_size == 4 and m.layers[0].blocks[1].attn_mask is not None
    m.eval()
    with torch.no_grad():
        y = m(torch.rand(2, 1, 251, 64))
    assert y.shape == (2, 768)


@needs_golden
def test_trim_matches_librosa():
    g = torch.load(GOLDEN / "features.pt")
    for name in ("icbhi", "sprs", "synth"):
        s, e = librosa_trim_indices(g[f"wav_{name}"], 1600, 800)
        assert [s, e] == g["trim_indices"][name], name
    mel = OperaBatchMel()
    with pytest.raises(RuntimeError):
        mel(g["wav_synth"])  # the original dies on 2-D padding -> explicit error


@needs_golden
def test_batch_mel_matches_reference():
    g = torch.load(GOLDEN / "features.pt")
    mel = OperaBatchMel()
    for name in ("icbhi", "sprs"):
        ours = mel(g[f"wav_{name}"])
        ref = g[f"feat_{name}"]
        assert ours.shape == ref.shape == (6, 64, 251)
        assert torch.allclose(ours, ref, atol=2e-3), (name, (ours - ref).abs().max())
    # batch-dependent normalization: feeding only the first 3 changes the values, which also match the original
    half = mel(g["wav_icbhi"][:3])
    assert torch.allclose(half, g["feat_icbhi_half"], atol=2e-3)
    assert not torch.allclose(half, g["feat_icbhi"][:3], atol=1e-3)


@needs_golden
def test_specaug_batch_shared_masks_match_reference():
    g = torch.load(GOLDEN / "features.pt")
    model = OperaCT(pretrained_path=None)
    torch.manual_seed(g["specaug_seed"])
    x = g["feat_icbhi"]
    v1 = model.timemask(model.freqmask(x.clone()))
    v2 = model.timemask(model.freqmask(x.clone()))
    assert torch.equal(v1, g["specaug_view1"]) and torch.equal(v2, g["specaug_view2"])
    # batch-shared mask (torchaudio default iid_masks=False): one mask position for the whole batch
    assert model.freqmask.iid_masks is False and model.timemask.iid_masks is False
    changed = (v1 != x)
    assert torch.equal(changed[0], changed[-1] & (x[0] != 0)) or torch.equal(changed[0] & (x[-1] != 0), changed[-1] & (x[0] != 0))


@needs_golden
@needs_ckpt
def test_htsat_checkpoint_and_embeddings_match_reference():
    g = torch.load(GOLDEN / "htsat.pt")
    m = HTSATSwin()
    dropped = load_opera_ct_checkpoint(m, str(CKPT))
    assert len(dropped) == 8 and set(g["unexpected_keys"]) <= set(dropped) and g["missing_keys"] == []
    ours = {k: list(v.shape) for k, v in m.state_dict().items()}
    assert ours == g["htsat_keys"]
    assert sum(p.numel() for p in m.parameters()) == g["htsat_n_params"]
    f = torch.load(GOLDEN / "features.pt")
    m.eval()
    with torch.no_grad():
        emb = m(f["feat_icbhi"].transpose(-2, -1).unsqueeze(1))
        emb_s = m(f["feat_sprs"].transpose(-2, -1).unsqueeze(1))
    assert torch.allclose(emb, g["emb_icbhi"], atol=1e-4), (emb - g["emb_icbhi"]).abs().max()
    assert torch.allclose(emb_s, g["emb_sprs"], atol=1e-4)
    # train mode: bn0 batch statistics + DropPath (same seed -> same random consumption order)
    for k, v in g["bn0_before"].items():
        assert torch.equal(m.bn0.state_dict()[k], v)
    m.train()
    torch.manual_seed(g["train_seed"])
    with torch.no_grad():
        emb_t = m(f["feat_icbhi"].transpose(-2, -1).unsqueeze(1))
    assert torch.allclose(emb_t, g["emb_train"], atol=1e-4)
    for k, v in g["bn0_after"].items():
        assert torch.allclose(m.bn0.state_dict()[k].float(), v.float(), atol=1e-6), k


def _assert_state_meta(module, meta):
    sd = module.state_dict()
    assert list(sd.keys()) == list(meta.keys())
    for k, (shape, total, absmean) in meta.items():
        v = sd[k].double()
        assert list(v.shape) == shape, k
        assert abs(float(v.sum()) - total) <= 1e-6 * max(1.0, abs(total)) and abs(float(v.abs().mean()) - absmean) <= 1e-8, k


@needs_golden
def test_heads_match_reference():
    g = torch.load(GOLDEN / "head.pt")
    h = torch.load(GOLDEN / "htsat.pt")
    tokens = h["emb_icbhi"].unsqueeze(1)
    torch.manual_seed(g["tfm_seed"])
    tfm = TfmEncHead(embed_dim=768, output_dim=4, depth=4, heads=1, mlp_ratio=1).eval()
    _assert_state_meta(tfm, g["tfm_state_meta"])  # same seed and construction order -> identical initialization
    torch.manual_seed(g["mlp_seed"])
    mlp = MLPHead(768, (128,), 4).eval()
    _assert_state_meta(mlp, g["mlp_state_meta"])
    with torch.no_grad():
        assert torch.allclose(tfm(tokens), g["logits_tfm"], atol=1e-5)
        assert torch.allclose(mlp(tokens), g["logits_mlp"], atol=1e-5)


@needs_ckpt
def test_model_freeze_and_training_forward():
    m = OperaCT(pretrained_path=str(CKPT), n_cls=4)
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all(n.startswith("htsat.") for n in frozen) and all(n.startswith("head.") for n in trainable)
    wav = torch.rand(4, 128000) * 0.1
    m.train()
    logits, loss = m.training_forward(wav, torch.tensor([0, 1, 2, 3]), torch.nn.functional.cross_entropy)
    assert logits.shape == (4, 4) and torch.isfinite(loss)
    loss.backward()
    assert m.head.linear.weight.grad is not None and m.htsat.patch_embed.proj.weight.grad is None
