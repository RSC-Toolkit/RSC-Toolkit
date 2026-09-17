"""
Official-code parity golden and configuration tests for the MVST (ICASSP 2024) group.

Golden source: scripts/reproduction/dump_mvst_goldens.py — the official {view}/models/ast.py run in the reference environment (timm 0.4.5)
(pos_embed and patch proj, which stay random, use det_fill on both sides). The AudioSet 16×16 ckpt is the HF mirror converted to the original key format
(scripts/reproduction/convert_hf_ast_to_yuangong.py, 5e-5 vs HF logits). Evaluation record of the authors' view ckpts: tests/golden/mvst/anchor_eval_record.json.
"""
import json
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.methods.mvst_fusion import FuseMLP, class_ratio_weights


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "mvst"
AS16 = REPO / "pretrained_models" / "upstream" / "audioset_16_16_0.4422.pth"
AUTHOR = REPO / "pretrained_models" / "upstream" / "mvst"
needs_golden = pytest.mark.skipif(not (GOLDEN / "features_input.pt").exists(), reason="golden missing")
needs_as16 = pytest.mark.skipif(not AS16.exists(), reason="audioset_16_16_0.4422.pth missing (convert_hf_ast_to_yuangong.py)")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


@needs_golden
@needs_as16
@pytest.mark.parametrize("view", [16, 32, 64, 128, 256])
def test_view_model_matches_reference(view):
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    m = MVSTViewAST(n_cls=4, view=view)
    _det_fill(m.v.pos_embed); _det_fill(m.v.patch_embed.proj.weight); _det_fill(m.v.patch_embed.proj.bias)
    m.eval()
    x = torch.load(GOLDEN / "features_input.pt")
    with torch.no_grad():
        f = m.forward_features(x)
    ref = torch.load(GOLDEN / f"view{view}_features.pt")
    assert torch.allclose(f, ref, atol=1e-4), float((f - ref).abs().max())
    shapes = json.load(open(GOLDEN / "state_dict_shapes.json"))[str(view)]
    strip = lambda d: {k: s for k, s in d.items() if not k.startswith(("v.head", "head"))}  # noqa: E731
    ours = strip({k: list(t.shape) for k, t in m.state_dict().items() if not k.startswith("classifier.")})
    assert ours == strip(shapes)
    assert tuple(m.v.pos_embed.shape) == (1, 1026, 768) and m.grid[0] * m.grid[1] == 1024


@needs_as16
def test_audioset_matching_key_load_semantics():
    """Original quirk: pos_embed (514≠1026) is not loaded for any view; patch proj is loaded only for the 16×16 view."""
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    m16 = MVSTViewAST(n_cls=4, view=16)
    info16 = m16.load_audioset_matching_keys(None)
    assert "pos_embed" in info16["mismatch"] and "patch_embed.proj.weight" in info16["loaded"]
    m32 = MVSTViewAST(n_cls=4, view=32)
    info32 = m32.load_audioset_matching_keys(None)
    assert "patch_embed.proj.weight" in info32["mismatch"] and "blocks.0.attn.qkv.weight" in info32["loaded"]


@pytest.mark.skipif(not (AUTHOR / "16" / "best.pth").exists(), reason="author MVST ckpt missing")
def test_author_checkpoints_load_strict():
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    for view in (16, 256):  # 16 = 'v.*' keys, 256 = 'module.v.*' keys (DataParallel)
        m = MVSTViewAST(n_cls=4, view=view, audioset_pretrain=False)
        m.load_reference_checkpoint(AUTHOR / str(view) / "best.pth")


def test_to_model_input_resizes_and_forward_shape():
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    m = MVSTViewAST(n_cls=4, view=64, audioset_pretrain=False).eval()
    x = torch.randn(2, 128, 798)  # toolkit fbank layout
    xin = m.to_model_input(x)
    assert xin.shape == (2, 1, 1024, 256)
    with torch.no_grad():
        assert m(x).shape == (2, 4)


def test_fusemlp_layout_and_forward():
    torch.manual_seed(0)
    f = FuseMLP(num_views=5, num_classes=4)
    assert sum(p.numel() for p in f.parameters()) == 5 * 768 * 768 + 768 * 4 + 4  # 2,952,196
    assert torch.equal(f.fc.bias, torch.zeros(4)) and float(f.fc.weight.std()) < 0.02
    xs = [torch.randn(3, 768) for _ in range(5)]
    assert f(*xs).shape == (3, 4)
    ref = f.fc(sum(torch.sigmoid(x @ getattr(f, f"W{i}")) * x for i, x in enumerate(xs, 1)))
    assert torch.allclose(f(*xs), ref)


def test_class_ratio_weights_are_proportional_quirk():
    w = class_ratio_weights(torch.tensor([0, 0, 0, 1, 2, 3]), 4)
    assert torch.allclose(w, torch.tensor([0.5, 1 / 6, 1 / 6, 1 / 6]))


def test_mvst_seed1_split_file_counts():
    """random.Random(1).shuffle over 920 recordings -> train 552 / test 368 (same as the author-ckpt evaluation record)."""
    import random

    p = REPO / "rsc_toolkit" / "datasets" / "icbhi" / "data" / "mvst_seed1_split.txt"
    lines = [l.split() for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 920
    assert sum(1 for _, s in lines if s == "train") == 552
    stems = sorted(r for r, _ in lines)
    idx = list(range(len(stems))); random.Random(1).shuffle(idx)
    train = {stems[i] for i in idx[: int(len(idx) * 0.6)]}
    assert all((s == "train") == (r in train) for r, s in lines)
