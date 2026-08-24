"""
PatchMixAST 의 P0(구조 일치) 골든 테스트.

골든 파일은 공식 저장소 코드(timm 0.4.5, torch 1.13)에서
scripts/reproduction/dump_patchmix_goldens.py 로 떴다. 이 테스트가 고정하는 것:
- state_dict 키·모양이 참조 ASTModel 과 동일 (추가는 classifier.* 만)
- 같은 입력에 대한 forward_features 출력 일치 (실측 max|diff| 3.8e-6)
- PatchMixLoss / PatchMixConLoss 값이 참조와 일치 (실측 오차 0)
- 뒤집힌 입력 차원 quirk 로 생기는 pos_embed 모양 (1, 2+79*12, 768)

AudioSet ckpt(약 350MB, pretrained_models/upstream/)가 필요해서 없으면 스킵한다.
"""
import json
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from rsc_toolkit.methods.patchmix import PatchMixConLoss, PatchMixLoss
from rsc_toolkit.models.ast_patchmix import AUDIOSET_CKPT_CANDIDATES


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "patchmix"
CKPT_AVAILABLE = any((REPO / c).exists() for c in AUDIOSET_CKPT_CANDIDATES)

needs_golden = pytest.mark.skipif(
    not (GOLDEN / "features_golden.pt").exists(),
    reason="골든 파일 없음 — scripts/reproduction/dump_patchmix_goldens.py 로 생성",
)
needs_ckpt = pytest.mark.skipif(
    not CKPT_AVAILABLE,
    reason="AudioSet AST ckpt 없음 (pretrained_models/upstream/audioset_0.4593.pth)",
)


@pytest.fixture(scope="module")
def model():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_patchmix")(n_cls=4)
    m.eval()
    return m


@needs_ckpt
@needs_golden
def test_state_dict_matches_reference(model):
    golden = json.load(open(GOLDEN / "state_dict_shapes.json"))
    ours = {k: list(v.shape) for k, v in model.state_dict().items()}

    missing = [k for k in golden if k not in ours]
    mismatch = {k: (golden[k], ours[k]) for k in golden if k in ours and golden[k] != ours[k]}
    assert not missing, f"참조에 있는 키가 없다: {missing[:10]}"
    assert not mismatch, f"모양 불일치: {dict(list(mismatch.items())[:5])}"

    extra = [k for k in ours if k not in golden and not k.startswith("classifier.")]
    assert not extra, f"참조에 없는 키(classifier.* 외): {extra[:10]}"


@needs_ckpt
@needs_golden
def test_forward_features_match_reference(model):
    x = torch.load(GOLDEN / "features_input.pt")
    ref = torch.load(GOLDEN / "features_golden.pt")
    with torch.no_grad():
        got = model.forward_features(x)
    # 실측(2026-08-24, torch 1.13 참조 vs torch 2.7): max|diff| 3.8e-6
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (
        f"forward 출력이 참조와 다르다: max|diff|={float((got - ref).abs().max()):.3e}"
    )


@needs_ckpt
def test_pos_embed_reproduces_swapped_dim_quirk(model):
    """
    원본 main.py 의 뒤집힌 input_fdim/tdim(798/128)으로 pos_embed 가
    (79x12) 격자 기준으로 만들어진다 — 2 + 948 = 950 토큰.
    이 quirk 는 논문 가중치·수치의 전제이므로 '고치면' 안 된다.
    """
    assert tuple(model.v.pos_embed.shape) == (1, 950, 768)


@needs_ckpt
def test_classifier_starts_as_copy_of_mlp_head(model):
    """원본 set_model: classifier = deepcopy(model.mlp_head)."""
    for (ka, pa), (kb, pb) in zip(
        model.mlp_head.state_dict().items(), model.classifier.state_dict().items()
    ):
        assert ka == kb
        assert torch.equal(pa, pb)
    # 별개 객체여야 한다 (같이 학습되면 안 됨)
    assert model.classifier is not model.mlp_head
    assert model.classifier[1].weight is not model.mlp_head[1].weight


@needs_ckpt
def test_patch_mix_replaces_patches_and_recomputes_lam(model):
    torch.manual_seed(0)
    np.random.seed(0)

    image = torch.arange(4 * 10 * 3, dtype=torch.float32).reshape(4, 10, 3)
    orig = image.clone()
    target = torch.tensor([0, 1, 2, 3])

    mixed, y_a, y_b, lam, index = model.patch_mix(image.clone(), target)

    assert torch.equal(y_a, target)
    assert torch.equal(y_b, target[index])
    # lam 은 실제 교체 비율로 재계산된다: 1 - num_mask/num_patch
    num_mask = round((1 - lam) * 10)
    assert abs(lam - (1 - num_mask / 10)) < 1e-9
    # 각 위치는 자기 것 그대로거나 donor(index) 의 같은 위치 패치다
    for b in range(4):
        for pos in range(10):
            row = mixed[b, pos]
            assert torch.equal(row, orig[b, pos]) or torch.equal(row, orig[index[b], pos])


@needs_golden
def test_losses_match_reference_goldens():
    g = torch.load(GOLDEN / "loss_goldens.pt")

    w = torch.tensor(g["class_nums"], dtype=torch.float32)
    w = 1.0 / (w / w.sum())
    w /= w.sum()

    got_u = float(PatchMixLoss(nn.CrossEntropyLoss())(g["pred"], g["y_a"], g["y_b"], g["lam"]))
    got_w = float(PatchMixLoss(nn.CrossEntropyLoss(weight=w))(g["pred"], g["y_a"], g["y_b"], g["lam"]))
    got_all = float(PatchMixConLoss(0.06, "all")(g["proj1"], g["proj2"], g["y_a"], g["y_b"], g["lam"], g["index"]))
    got_diff = float(
        PatchMixConLoss(0.06, "diff_label")(g["proj1"], g["proj2"], g["y_a"], g["y_b"], g["lam"], g["index"])
    )

    assert got_u == pytest.approx(g["patchmix_loss_unweighted"], abs=1e-6)
    assert got_w == pytest.approx(g["patchmix_loss_weighted"], abs=1e-6)
    assert got_all == pytest.approx(g["patchmix_con_loss_all"], abs=1e-6)
    assert got_diff == pytest.approx(g["patchmix_con_loss_diff_label"], abs=1e-6)


def test_patchmix_con_loss_rejects_unknown_negative_pair():
    with pytest.raises(ValueError, match="negative_pair"):
        PatchMixConLoss(0.06, "bogus")
