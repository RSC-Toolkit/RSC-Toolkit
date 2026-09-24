"""
LungAdapter (Interspeech 2024) — verification of a re-implementation from the paper description (no official code -> no golden comparison).

Pinned:
- table of trainable/total parameter counts per flag combination (paper Table 1/2: Attn only 0.34M/87.87M, Res only 2.14M/89.67M, both 2.48M/90.01M, ±0.01M).
  The default combination (2D DWConv + vector s1/s2 + Res 115 bias, res_scale none = the paper's equations as written) is within ±0.01M for Attn only and Res only,
  but both gives 2.468M, 0.0116M off the paper's 2.48 (unresolved). Among the 6 enumerated combinations only res_scale=vector satisfies all three values at once
  (an invented parameter, so not the default). Other combinations not enumerated (e.g. LN, no bias) also satisfy all three values, so the parameter count cannot pin down the structure.
- adapter module shapes, residuals, freezing, det_fill forward determinism.
- adapter identity: with W_up=0 (default zero_init_up) logits equal the original PatchMixAST (both attn_insert modes).
- 2D DWConv path = forward match with the original Mona MonaOp (tests/golden/lungadapter/mona_op_ref.py, MIT) (golden = direct execution of the reference code).
"""
import importlib.util
import pathlib
import re

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.models.ast_lungadapter import AttentionAdapter, LungAdapterAST, LungAdapterBlock, MultiScaleDWConv, ResAdapter
from rsc_toolkit.models.ast_patchmix import AUDIOSET_CKPT_CANDIDATES, PatchMixAST


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
MONA_REF = REPO / "tests/golden/lungadapter/mona_op_ref.py"
CKPT_AVAILABLE = any((REPO / c).exists() for c in AUDIOSET_CKPT_CANDIDATES)
needs_ckpt = pytest.mark.skipif(not CKPT_AVAILABLE, reason="AudioSet AST ckpt missing (pretrained_models/upstream/audioset_0.4593.pth)")
needs_mona = pytest.mark.skipif(not MONA_REF.exists(), reason="Mona reference code missing (tests/golden/lungadapter/mona_op_ref.py)")

SCRATCH = dict(audioset_pretrain=False, imagenet_pretrain=False)   # skeleton only (for parameter count/structure tests, no download)

# paper (Trainable M, Total M) — Total is the paper's 87.53 plus the rounded Trainable
PAPER = {"attn_only": (0.34, 87.87), "res_only": (2.14, 89.67), "both": (2.48, 90.01)}
# trainable parameter counts per (conv_dim, scale_form, res_scale) combination — among the 6 enumerated, only res_scale=vector satisfies all three tables within ±0.01M
DEFAULT_COMBO = ("2d", "vector", "none")        # the paper's equations as written (default)
FIT_COMBO = ("2d", "vector", "vector")          # matches all three paper-reported values (768-d res_scale, an invented parameter)
COMBOS = {
    ("2d", "vector", "none"):   {"attn_only": 338_116, "res_only": 2_134_888, "both": 2_468_392},   # default
    ("2d", "vector", "vector"): {"attn_only": 338_116, "res_only": 2_144_104, "both": 2_477_608},
    ("2d", "scalar", "vector"): {"attn_only": 328_732, "res_only": 2_144_104, "both": 2_468_224},
    ("2d", "vector", "scalar"): {"attn_only": 338_116, "res_only": 2_134_900, "both": 2_468_404},
    ("1d", "vector", "vector"): {"attn_only": 325_060, "res_only": 2_144_104, "both": 2_464_552},
    ("1d", "scalar", "none"):   {"attn_only": 315_676, "res_only": 2_134_888, "both": 2_445_952},
}
PATCHMIX_TOTAL = 87_531_736  # total of PatchMixAST(n_cls=4) (including classifier, mlp_head, dead timm head) = paper 87.53M


def _det_fill(t, amp: float = 0.5):
    with torch.no_grad():
        t.copy_(torch.linspace(-amp, amp, t.numel()).reshape(t.shape))


@pytest.fixture(scope="module")
def scratch_default():
    m = LungAdapterAST(**SCRATCH)
    m.eval()
    return m


# ------------------------------------------------------------------ adapter modules
def test_attention_adapter_shapes_and_residual():
    ad = AttentionAdapter(dim=768, hidden=16, conv_dim="2d", zero_init_up=True)
    x = torch.randn(2, 2 + 12 * 79, 768)
    y = ad(x, grid=(12, 79))
    assert y.shape == x.shape
    assert torch.allclose(y, x)                       # W_up=0, s1=1 -> y = s1·X = X (identity)
    nn.init.normal_(ad.up.weight, std=0.02)
    assert not torch.allclose(ad(x, grid=(12, 79)), x)
    with pytest.raises(ValueError):
        ad(x, grid=(12, 80))                          # token count mismatch
    with pytest.raises(ValueError):
        ad(x, grid=None)                              # 2D requires a grid


def test_attention_adapter_grid_orientation_matters():
    ad = AttentionAdapter(dim=32, hidden=4, conv_dim="2d", zero_init_up=False)
    for p in ad.parameters():
        _det_fill(p)
    x = torch.linspace(-1, 1, 2 * (2 + 12 * 79) * 32).reshape(2, 2 + 12 * 79, 32)
    y1, y2 = ad(x, grid=(12, 79)), ad(x, grid=(79, 12))
    assert not torch.allclose(y1, y2)                 # grid orientation (freq 12 × time 79) changes the result -> the model must pass the real grid


def test_attention_adapter_1d_and_scalar():
    ad = AttentionAdapter(dim=64, hidden=8, conv_dim="1d", scale_form="scalar", zero_init_up=False)
    assert ad.s1.dim() == 0 and ad.s2.dim() == 0
    x = torch.randn(3, 50, 64)
    assert ad(x).shape == x.shape
    assert sum(p.numel() for p in ad.conv.dwconvs.parameters()) == 8 * (3 + 5 + 7) + 3 * 8


def test_multiscale_dwconv_formula():
    conv = MultiScaleDWConv(4, "2d")
    h = torch.randn(1, 4, 5, 6)
    hp = h + (conv.dwconvs[0](h) + conv.dwconvs[1](h) + conv.dwconvs[2](h)) / 3.0
    assert torch.allclose(conv(h), conv.projector(hp) + hp, atol=1e-6)


def test_res_adapter_bottleneck_and_identity():
    ad = ResAdapter(768, 0.15)
    assert ad.bottleneck == 115 and ad.down.weight.shape == (115, 768)
    assert ResAdapter(768, 0.15, bottleneck=116).bottleneck == 116
    x = torch.randn(2, 10, 768)
    assert torch.equal(ad(x), torch.zeros_like(x))    # W_up=0
    assert sum(p.numel() for p in ad.parameters()) == 768 * 115 + 115 + 115 * 768 + 768


def test_block_flags_validation():
    blk = nn.Module()
    with pytest.raises(ValueError):
        LungAdapterBlock(blk, None, None, attn_insert="middle")
    with pytest.raises(ValueError):
        LungAdapterBlock(blk, None, None, res_input="ln")
    with pytest.raises(ValueError):
        LungAdapterBlock(blk, None, None, res_scale="huge")


# --------------------------------------------------------------- parameter count table
@pytest.mark.parametrize("combo", sorted(COMBOS))
def test_parameter_table(combo):
    conv_dim, scale_form, res_scale = combo
    for variant, expected in COMBOS[combo].items():
        m = LungAdapterAST(
            use_attn_adapter=variant != "res_only", use_res_adapter=variant != "attn_only",
            conv_dim=conv_dim, scale_form=scale_form, res_scale=res_scale, **SCRATCH,
        )
        tr, tot = m.count_parameters()
        assert tr == expected, (combo, variant, tr)
        assert tot == PATCHMIX_TOTAL + tr - 4_612               # classifier (4,612) is already included in the PatchMixAST total
        if combo == FIT_COMBO:
            # fitted combination: within ±0.01M of the paper table (Total is the paper's 87.53 + rounded Trainable, so it differs from ours by ≤0.006M)
            assert abs(tr / 1e6 - PAPER[variant][0]) <= 0.01 + 1e-9
            assert abs(tot / 1e6 - PAPER[variant][1]) <= 0.01 + 1e-9
        if combo == DEFAULT_COMBO:
            # default (paper equations) combination: attn_only/res_only within ±0.01M, both is 0.0116M short (documented unresolved gap)
            gap = PAPER[variant][0] - tr / 1e6
            assert abs(gap) <= 0.01 + 1e-9 if variant != "both" else abs(gap - 0.0116) < 1e-3


def test_default_is_paper_formula_and_only_fit_combo_matches_all_three():
    """The default is the paper's equations (res_input x, res_scale none). Within the enumerated COMBOS, only FIT_COMBO matches all three paper-reported values at once
    (not a uniqueness claim over the whole design space; see the module docstring)."""
    m = LungAdapterAST(**SCRATCH)
    assert m.adapter_cfg["res_input"] == "x" and m.adapter_cfg["res_scale"] == "none"
    assert (m.adapter_cfg["conv_dim"], m.adapter_cfg["scale_form"], m.adapter_cfg["res_scale"]) == DEFAULT_COMBO
    ok = [c for c, t in COMBOS.items() if all(abs(t[v] / 1e6 - PAPER[v][0]) <= 0.01 + 1e-9 for v in PAPER)]
    assert ok == [FIT_COMBO]


# -------------------------------------------------------------------- freezing/determinism
def test_freeze_only_adapters_and_classifier(scratch_default):
    m = scratch_default
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith("classifier.") or (".attn_adapter." in n or ".res_adapter." in n or n.endswith(".res_scale")) for n in trainable)
    assert all(not p.requires_grad for n, p in m.named_parameters() if ".block." in n)   # original timm Block
    assert all(not p.requires_grad for p in m.v.patch_embed.parameters()) and not m.v.pos_embed.requires_grad
    assert all(not p.requires_grad for p in m.mlp_head.parameters())
    assert len(list(m.adapter_parameters())) == 12 * (2 + 2 + 6 + 2 + 2 + 4)   # s1,s2 / down w,b / dwconv×3 w,b / proj w,b / up w,b / res down·up w,b (res_scale none -> buffer)
    assert not isinstance(m.v.blocks[0].res_scale, nn.Parameter) and float(m.v.blocks[0].res_scale) == 1.0
    mv = LungAdapterAST(res_scale="vector", **SCRATCH)
    assert len(list(mv.adapter_parameters())) == 12 * (2 + 2 + 6 + 2 + 2 + 4 + 1)
    assert m.token_grid == (12, 79)


def test_forward_deterministic_after_det_fill():
    m = LungAdapterAST(**SCRATCH)
    for p in m.parameters():
        _det_fill(p, amp=0.05)   # filling all parameters with ±0.5 overflows the 12-block residual stream (res_input=x, unnormalized) -> NaN. Only determinism is checked
    m.eval()
    x = torch.linspace(-1, 1, 128 * 798).reshape(1, 128, 798)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b) and a.shape == (1, 4) and torch.isfinite(a).all()
    # pin that the forward path passed the real patch grid (freq 12 × time 79) to the blocks
    assert all(blk.grid == (12, 79) for blk in m.v.blocks)


def test_training_forward_signature(scratch_default):
    x = torch.randn(2, 128, 798)
    y = torch.tensor([0, 2])
    logits, loss = scratch_default.training_forward(x, y, nn.CrossEntropyLoss(), meta_labels=None)
    assert logits.shape == (2, 4) and loss.dim() == 0


# ----------------------------------------------------------------- adapter identity
@pytest.mark.parametrize("attn_insert", ["pre_residual", "post_residual"])
@pytest.mark.parametrize("res_input", ["x", "norm2"])
def test_zero_adapters_equal_patchmix_ast(attn_insert, res_input):
    torch.manual_seed(0)
    ref = PatchMixAST(n_cls=4, method="ce", input_layout="freq_time", **SCRATCH)
    m = LungAdapterAST(attn_insert=attn_insert, res_input=res_input, zero_init_up=True, **SCRATCH)
    # match the frozen parts (v.*, classifier) to the reference model (adapter parameters do not affect the output since W_up is 0)
    v_state = {k: v for k, v in ref.v.state_dict().items()}
    ours_v = {}
    for k, v in m.v.state_dict().items():
        kk = re.sub(r"^blocks\.(\d+)\.block\.", r"blocks.\1.", k)
        if kk in v_state:
            ours_v[k] = v_state[kk]
    missing = m.v.load_state_dict(ours_v, strict=False)
    assert not [k for k in missing.missing_keys if ".block." in k or not k.startswith("blocks.")]
    m.classifier.load_state_dict(ref.classifier.state_dict())
    for blk in m.v.blocks:                         # also make random adapter parameters deterministic (keep W_up=0, s1=s2=1)
        for n, p in blk.attn_adapter.named_parameters():
            if not (n.startswith("up.") or n in {"s1", "s2"}):
                _det_fill(p)
        _det_fill(blk.res_adapter.down.weight)
    ref.eval(); m.eval()
    x = torch.randn(2, 128, 798)
    with torch.no_grad():
        assert torch.allclose(ref(x), m(x), atol=1e-5)
    # filling W_up changes the output (the adapter is actually on the path)
    for blk in m.v.blocks:
        nn.init.normal_(blk.attn_adapter.up.weight, std=0.02)
    with torch.no_grad():
        assert not torch.allclose(ref(x), m(x), atol=1e-5)


# ---------------------------------------------------------- comparison with original Mona (golden = reference code execution)
def _load_mona_op():
    spec = importlib.util.spec_from_file_location("mona_op_ref", MONA_REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MonaOp


@needs_mona
def test_2d_conv_path_matches_mona_op():
    MonaOp = _load_mona_op()
    torch.manual_seed(1)
    ref = MonaOp(16)
    ours = MultiScaleDWConv(16, "2d")
    ours.dwconvs[0].load_state_dict(ref.conv1.state_dict())
    ours.dwconvs[1].load_state_dict(ref.conv2.state_dict())
    ours.dwconvs[2].load_state_dict(ref.conv3.state_dict())
    ours.projector.load_state_dict(ref.projector.state_dict())
    h = torch.randn(2, 16, 12, 79)
    assert torch.allclose(ours(h), ref(h), atol=1e-6)


@needs_mona
def test_attention_adapter_patch_tokens_match_mona_layout():
    """Same token<->grid arrangement as reshape(b,h,w,c).permute(0,3,1,2) in Mona.forward (excluding cls/dist, s=1, no LN)."""
    MonaOp = _load_mona_op()
    torch.manual_seed(2)
    ad = AttentionAdapter(dim=32, hidden=16, conv_dim="2d", n_prefix_tokens=0, zero_init_up=False)
    ref = MonaOp(16)
    ref.conv1.load_state_dict(ad.conv.dwconvs[0].state_dict()); ref.conv2.load_state_dict(ad.conv.dwconvs[1].state_dict())
    ref.conv3.load_state_dict(ad.conv.dwconvs[2].state_dict()); ref.projector.load_state_dict(ad.conv.projector.state_dict())
    x = torch.randn(2, 12 * 79, 32)
    with torch.no_grad():
        p1 = ad.down(x)
        b, n, c = p1.shape
        z = ref(p1.reshape(b, 12, 79, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, n, c)
        expected = ad.up(ad.s2 * torch.nn.functional.gelu(z)) + x
        assert torch.allclose(ad(x, grid=(12, 79)), expected, atol=1e-6)


# ------------------------------------------------------------ total parameters with the real ckpt
@needs_ckpt
def test_default_matches_paper_counts_with_ckpt():
    m = LungAdapterAST(n_cls=4)
    tr, tot = m.count_parameters()
    assert tr == 2_468_392 and tot == 89_995_516           # default (paper equations): 0.0116M off the paper's 2.48M/90.01M (documented)
    assert list(m.v.pos_embed.shape) == [1, 2 + 79 * 12, 768]
    mv = LungAdapterAST(n_cls=4, res_scale="vector")
    tr, tot = mv.count_parameters()
    assert tr == 2_477_608 and tot == 90_004_732
    assert abs(tr / 1e6 - 2.48) <= 0.01 and abs(tot / 1e6 - 90.01) <= 0.01
