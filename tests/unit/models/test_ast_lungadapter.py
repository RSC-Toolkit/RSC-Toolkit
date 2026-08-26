"""
LungAdapter(Interspeech 2024) — 논문 명세 재구현 검증 (공식 코드 없음 → 골든 대조 불가).

고정하는 것:
- 플래그 조합별 학습/전체 파라미터 수 표 (논문 Table 1/2: Attn만 0.34M/87.87M, Res만 2.14M/89.67M, 둘 다 2.48M/90.01M, ±0.01M).
  기본 조합(2D DWConv + 벡터 s1/s2 + Res 115 bias, res_scale none = 논문 식 그대로)은 Attn만·Res만 은 ±0.01M 안이지만
  둘 다 는 2.468M 으로 논문 2.48 과 0.0116M 차(미해결). 열거한 6개 조합 중 세 값을 동시에 만족하는 것은 res_scale=vector 뿐
  (발명 파라미터라 기본값 아님). (정찰 verdict.corrections: LN·무 bias 등 열거하지 않은 다른 조합도 세 값을 만족 → 파라미터 수로 구조 확정 불가.)
- 어댑터 모듈 형상·잔차·동결·det_fill forward 결정성.
- 어댑터 항등성: W_up=0(기본 zero_init_up) 이면 로짓이 원 PatchMixAST 와 동일 (attn_insert 두 방식 모두).
- 2D DWConv 경로 = Mona 원형 MonaOp(tests/golden/lungadapter/mona_op_ref.py, MIT) 와 forward 일치 (골든 = 참조 코드 직접 실행).
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
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
MONA_REF = REPO / "tests/golden/lungadapter/mona_op_ref.py"
CKPT_AVAILABLE = any((REPO / c).exists() for c in AUDIOSET_CKPT_CANDIDATES)
needs_ckpt = pytest.mark.skipif(not CKPT_AVAILABLE, reason="AudioSet AST ckpt 없음 (pretrained_models/upstream/audioset_0.4593.pth)")
needs_mona = pytest.mark.skipif(not MONA_REF.exists(), reason="Mona 참조 코드 없음 (tests/golden/lungadapter/mona_op_ref.py)")

SCRATCH = dict(audioset_pretrain=False, imagenet_pretrain=False)   # 골격만 (파라미터 수·구조 테스트용, 다운로드 없음)

# 논문 (Trainable M, Total M) — Total 은 논문 표기 87.53 + Trainable 의 반올림 합
PAPER = {"attn_only": (0.34, 87.87), "res_only": (2.14, 89.67), "both": (2.48, 90.01)}
# (conv_dim, scale_form, res_scale) 조합별 학습 파라미터 수 — 열거한 6개 중 세 표를 동시에 ±0.01M 로 만족하는 것은 res_scale=vector 뿐
DEFAULT_COMBO = ("2d", "vector", "none")        # 논문 식 그대로 (기본값)
FIT_COMBO = ("2d", "vector", "vector")          # 세 앵커 동시 일치 (768-d res_scale 발명 파라미터)
COMBOS = {
    ("2d", "vector", "none"):   {"attn_only": 338_116, "res_only": 2_134_888, "both": 2_468_392},   # 기본
    ("2d", "vector", "vector"): {"attn_only": 338_116, "res_only": 2_144_104, "both": 2_477_608},
    ("2d", "scalar", "vector"): {"attn_only": 328_732, "res_only": 2_144_104, "both": 2_468_224},
    ("2d", "vector", "scalar"): {"attn_only": 338_116, "res_only": 2_134_900, "both": 2_468_404},
    ("1d", "vector", "vector"): {"attn_only": 325_060, "res_only": 2_144_104, "both": 2_464_552},
    ("1d", "scalar", "none"):   {"attn_only": 315_676, "res_only": 2_134_888, "both": 2_445_952},
}
PATCHMIX_TOTAL = 87_531_736  # PatchMixAST(n_cls=4) 전체(classifier·mlp_head·timm 죽은 head 포함) = 논문 87.53M


def _det_fill(t, amp: float = 0.5):
    with torch.no_grad():
        t.copy_(torch.linspace(-amp, amp, t.numel()).reshape(t.shape))


@pytest.fixture(scope="module")
def scratch_default():
    m = LungAdapterAST(**SCRATCH)
    m.eval()
    return m


# ------------------------------------------------------------------ 어댑터 모듈
def test_attention_adapter_shapes_and_residual():
    ad = AttentionAdapter(dim=768, hidden=16, conv_dim="2d", zero_init_up=True)
    x = torch.randn(2, 2 + 12 * 79, 768)
    y = ad(x, grid=(12, 79))
    assert y.shape == x.shape
    assert torch.allclose(y, x)                       # W_up=0, s1=1 → y = s1·X = X (항등)
    nn.init.normal_(ad.up.weight, std=0.02)
    assert not torch.allclose(ad(x, grid=(12, 79)), x)
    with pytest.raises(ValueError):
        ad(x, grid=(12, 80))                          # 토큰 수 불일치
    with pytest.raises(ValueError):
        ad(x, grid=None)                              # 2D 는 격자 필수


def test_attention_adapter_grid_orientation_matters():
    ad = AttentionAdapter(dim=32, hidden=4, conv_dim="2d", zero_init_up=False)
    for p in ad.parameters():
        _det_fill(p)
    x = torch.linspace(-1, 1, 2 * (2 + 12 * 79) * 32).reshape(2, 2 + 12 * 79, 32)
    y1, y2 = ad(x, grid=(12, 79)), ad(x, grid=(79, 12))
    assert not torch.allclose(y1, y2)                 # 격자 방향(freq 12 × time 79)이 결과를 바꾼다 → 모델이 실제 격자를 넘겨야 함


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


# --------------------------------------------------------------- 파라미터 수 표
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
        assert tot == PATCHMIX_TOTAL + tr - 4_612               # classifier(4,612)는 PatchMixAST 총계에 이미 포함
        if combo == FIT_COMBO:
            # 맞춤 조합: 논문 표와 ±0.01M (Total 은 논문이 87.53+Trainable 반올림 합이라 우리 값과 ≤0.006M 차이)
            assert abs(tr / 1e6 - PAPER[variant][0]) <= 0.01 + 1e-9
            assert abs(tot / 1e6 - PAPER[variant][1]) <= 0.01 + 1e-9
        if combo == DEFAULT_COMBO:
            # 기본(논문 식) 조합: attn_only/res_only 는 ±0.01M 안, both 는 0.0116M 부족 (문서화된 미해결 차이)
            gap = PAPER[variant][0] - tr / 1e6
            assert abs(gap) <= 0.01 + 1e-9 if variant != "both" else abs(gap - 0.0116) < 1e-3


def test_default_is_paper_formula_and_only_fit_combo_matches_all_three():
    """기본값은 논문 식(res_input x, res_scale none). 열거한 COMBOS 안에서 세 앵커를 동시에 만족하는 것은 FIT_COMBO 뿐
    (전체 설계 공간에서의 유일성 주장이 아님 — verdict.corrections 참조)."""
    m = LungAdapterAST(**SCRATCH)
    assert m.adapter_cfg["res_input"] == "x" and m.adapter_cfg["res_scale"] == "none"
    assert (m.adapter_cfg["conv_dim"], m.adapter_cfg["scale_form"], m.adapter_cfg["res_scale"]) == DEFAULT_COMBO
    ok = [c for c, t in COMBOS.items() if all(abs(t[v] / 1e6 - PAPER[v][0]) <= 0.01 + 1e-9 for v in PAPER)]
    assert ok == [FIT_COMBO]


# -------------------------------------------------------------------- 동결·결정성
def test_freeze_only_adapters_and_classifier(scratch_default):
    m = scratch_default
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith("classifier.") or (".attn_adapter." in n or ".res_adapter." in n or n.endswith(".res_scale")) for n in trainable)
    assert all(not p.requires_grad for n, p in m.named_parameters() if ".block." in n)   # 원 timm Block
    assert all(not p.requires_grad for p in m.v.patch_embed.parameters()) and not m.v.pos_embed.requires_grad
    assert all(not p.requires_grad for p in m.mlp_head.parameters())
    assert len(list(m.adapter_parameters())) == 12 * (2 + 2 + 6 + 2 + 2 + 4)   # s1,s2 / down w,b / dwconv×3 w,b / proj w,b / up w,b / res down·up w,b (res_scale none → 버퍼)
    assert not isinstance(m.v.blocks[0].res_scale, nn.Parameter) and float(m.v.blocks[0].res_scale) == 1.0
    mv = LungAdapterAST(res_scale="vector", **SCRATCH)
    assert len(list(mv.adapter_parameters())) == 12 * (2 + 2 + 6 + 2 + 2 + 4 + 1)
    assert m.token_grid == (12, 79)


def test_forward_deterministic_after_det_fill():
    m = LungAdapterAST(**SCRATCH)
    for p in m.parameters():
        _det_fill(p, amp=0.05)   # ±0.5 로 전 파라미터를 채우면 12 블록 잔차 스트림(res_input=x, 비정규화)이 overflow → NaN. 결정성만 본다
    m.eval()
    x = torch.linspace(-1, 1, 128 * 798).reshape(1, 128, 798)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b) and a.shape == (1, 4) and torch.isfinite(a).all()
    # forward 경로가 실제 패치 격자 (freq 12 × time 79) 를 블록에 전달했는지 고정
    assert all(blk.grid == (12, 79) for blk in m.v.blocks)


def test_training_forward_signature(scratch_default):
    x = torch.randn(2, 128, 798)
    y = torch.tensor([0, 2])
    logits, loss = scratch_default.training_forward(x, y, nn.CrossEntropyLoss(), meta_labels=None)
    assert logits.shape == (2, 4) and loss.dim() == 0


# ----------------------------------------------------------------- 어댑터 항등성
@pytest.mark.parametrize("attn_insert", ["pre_residual", "post_residual"])
@pytest.mark.parametrize("res_input", ["x", "norm2"])
def test_zero_adapters_equal_patchmix_ast(attn_insert, res_input):
    torch.manual_seed(0)
    ref = PatchMixAST(n_cls=4, method="ce", input_layout="freq_time", **SCRATCH)
    m = LungAdapterAST(attn_insert=attn_insert, res_input=res_input, zero_init_up=True, **SCRATCH)
    # 동결 부분(v.*, classifier)을 참조 모델과 동일하게 맞춘다 (어댑터 파라미터는 W_up 이 0 이라 출력 무관)
    v_state = {k: v for k, v in ref.v.state_dict().items()}
    ours_v = {}
    for k, v in m.v.state_dict().items():
        kk = re.sub(r"^blocks\.(\d+)\.block\.", r"blocks.\1.", k)
        if kk in v_state:
            ours_v[k] = v_state[kk]
    missing = m.v.load_state_dict(ours_v, strict=False)
    assert not [k for k in missing.missing_keys if ".block." in k or not k.startswith("blocks.")]
    m.classifier.load_state_dict(ref.classifier.state_dict())
    for blk in m.v.blocks:                         # 어댑터 내부 난수 파라미터도 결정화 (W_up=0, s1=s2=1 은 유지)
        for n, p in blk.attn_adapter.named_parameters():
            if not (n.startswith("up.") or n in {"s1", "s2"}):
                _det_fill(p)
        _det_fill(blk.res_adapter.down.weight)
    ref.eval(); m.eval()
    x = torch.randn(2, 128, 798)
    with torch.no_grad():
        assert torch.allclose(ref(x), m(x), atol=1e-5)
    # W_up 을 채우면 달라진다 (어댑터가 실제로 경로에 있음)
    for blk in m.v.blocks:
        nn.init.normal_(blk.attn_adapter.up.weight, std=0.02)
    with torch.no_grad():
        assert not torch.allclose(ref(x), m(x), atol=1e-5)


# ---------------------------------------------------------- Mona 원형 대조 (골든 = 참조 코드 실행)
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
    """Mona.forward 의 reshape(b,h,w,c).permute(0,3,1,2) 토큰↔격자 배열과 동일한지 (cls/dist 제외, s=1, LN 없음)."""
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


# ------------------------------------------------------------ 실제 ckpt 기준 총 파라미터
@needs_ckpt
def test_default_matches_paper_counts_with_ckpt():
    m = LungAdapterAST(n_cls=4)
    tr, tot = m.count_parameters()
    assert tr == 2_468_392 and tot == 89_995_516           # 기본(논문 식): 논문 2.48M/90.01M 과 0.0116M 차 (문서화)
    assert list(m.v.pos_embed.shape) == [1, 2 + 79 * 12, 768]
    mv = LungAdapterAST(n_cls=4, res_scale="vector")
    tr, tot = mv.count_parameters()
    assert tr == 2_477_608 and tot == 90_004_732
    assert abs(tr / 1e6 - 2.48) <= 0.01 and abs(tot / 1e6 - 90.01) <= 0.01
