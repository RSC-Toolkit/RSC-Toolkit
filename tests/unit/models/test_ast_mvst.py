"""
MVST(ICASSP 2024) 묶음의 P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_mvst_goldens.py — 공식 {view}/models/ast.py 를 참조 환경(timm 0.4.5)에서 실행
(랜덤으로 남는 pos_embed·패치 proj 는 양쪽 det_fill). AudioSet 16×16 ckpt 는 HF 미러를 원본 키 형식으로 변환한 것
(scripts/reproduction/convert_hf_ast_to_yuangong.py, HF 로짓 대비 5e-5). 저자 뷰 ckpt 앵커 평가 기록: tests/golden/mvst/anchor_eval_record.json.
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
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "mvst"
AS16 = REPO / "pretrained_models" / "upstream" / "audioset_16_16_0.4422.pth"
AUTHOR = REPO / "pretrained_models" / "upstream" / "mvst"
needs_golden = pytest.mark.skipif(not (GOLDEN / "features_input.pt").exists(), reason="골든 없음")
needs_as16 = pytest.mark.skipif(not AS16.exists(), reason="audioset_16_16_0.4422.pth 없음 (convert_hf_ast_to_yuangong.py)")


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
    """원본 quirk: pos_embed(514≠1026)는 전 뷰 미로드, 패치 proj 는 16×16 뷰만 로드."""
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    m16 = MVSTViewAST(n_cls=4, view=16)
    info16 = m16.load_audioset_matching_keys(None)
    assert "pos_embed" in info16["mismatch"] and "patch_embed.proj.weight" in info16["loaded"]
    m32 = MVSTViewAST(n_cls=4, view=32)
    info32 = m32.load_audioset_matching_keys(None)
    assert "patch_embed.proj.weight" in info32["mismatch"] and "blocks.0.attn.qkv.weight" in info32["loaded"]


@pytest.mark.skipif(not (AUTHOR / "16" / "best.pth").exists(), reason="저자 MVST ckpt 없음")
def test_author_checkpoints_load_strict():
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    for view in (16, 256):  # 16 = 'v.*' 키, 256 = 'module.v.*' 키 (DataParallel)
        m = MVSTViewAST(n_cls=4, view=view, audioset_pretrain=False)
        m.load_reference_checkpoint(AUTHOR / str(view) / "best.pth")


def test_to_model_input_resizes_and_forward_shape():
    from rsc_toolkit.models.ast_mvst import MVSTViewAST

    m = MVSTViewAST(n_cls=4, view=64, audioset_pretrain=False).eval()
    x = torch.randn(2, 128, 798)  # 툴킷 fbank 레이아웃
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
    """random.Random(1).shuffle 로 920 녹음 → train 552 / test 368 (정찰·앵커 평가와 동일)."""
    import random

    p = REPO / "rsc_toolkit" / "datasets" / "icbhi" / "data" / "mvst_seed1_split.txt"
    lines = [l.split() for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 920
    assert sum(1 for _, s in lines if s == "train") == 552
    stems = sorted(r for r, _ in lines)
    idx = list(range(len(stems))); random.Random(1).shuffle(idx)
    train = {stems[i] for i in idx[: int(len(idx) * 0.6)]}
    assert all((s == "train") == (r in train) for r, s in lines)
