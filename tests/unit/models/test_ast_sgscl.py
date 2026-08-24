"""
SG-SCL 방법(ICASSP 2024)의 P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_sgscl_method_goldens.py
- ref 부분: 참조 환경(torch 1.13 + timm 0.4.5)에서 공식 sgscl 코드로 생성.
- pipeline 부분: 공식 util/augmentation.py 를 우리 venv 에서 실행해 생성
  (RNG 소모가 numpy/python/torch 인덱싱뿐이라 버전 비의존).

실측(2026-08-24): MetaCL·합성 손실 diff 0.0, 두 뷰 누적 마스킹 torch.equal,
AST state_dict 159키 동형, forward max|diff| 3.8e-06.
"""
import json
import pathlib
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn

from rsc_toolkit.augmentations.specaugment import SpecAugment
from rsc_toolkit.methods.sgscl import (
    MetaCL,
    apply_specaug_paper,
    sgscl_projections,
    sgscl_training_losses,
)
from rsc_toolkit.models.grl import ReverseLayerF
from rsc_toolkit.models.projector import Projector


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "sgscl_method"
AUDIOSET = any((REPO / c).exists() for c in (
    "pretrained_models/upstream/audioset_0.4593.pth", "pretrained_models/audioset_10_10_0.4593.pth"))

needs_golden = pytest.mark.skipif(
    not (GOLDEN / "metacl_golden.pt").exists(),
    reason="골든 없음 — scripts/reproduction/dump_sgscl_method_goldens.py 로 생성",
)
needs_ast = pytest.mark.skipif(not AUDIOSET, reason="AudioSet AST ckpt 없음")


def _det_fill(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


# ------------------------------------------------------------------ 손실 골든
@needs_golden
def test_metacl_matches_reference():
    g = torch.load(GOLDEN / "metacl_golden.pt")
    mcl = MetaCL(temperature=0.06)
    assert torch.equal(mcl(g["proj1"], g["proj2"], g["meta"]), g["loss_meta"])
    assert torch.equal(mcl(g["proj1"], g["proj2"], None), g["loss_none"])


@needs_golden
def test_sgscl_projection_composition_matches_reference():
    """Projector + target_type='project1_project2block' + MetaCL 합성 (원본 main.py:409-417)."""
    c = torch.load(GOLDEN / "sgscl_composition_golden.pt")
    proj = Projector(16, 16)
    for _, p in sorted(proj.named_parameters()):
        _det_fill(p)
    proj.eval()

    p1, p2 = sgscl_projections(c["feat1"], c["feat2"], proj, "project1_project2block")
    assert torch.equal(p1.detach(), c["p1"])
    assert torch.equal(p2, c["p2"])
    assert p1.requires_grad and not p2.requires_grad  # stop-grad 는 두 번째 뷰에만
    assert torch.equal(MetaCL(temperature=0.06)(p1, p2, c["meta"]), c["loss"])


def test_sgscl_projections_unknown_target_type_raises():
    with pytest.raises(ValueError, match="target_type"):
        sgscl_projections(torch.zeros(2, 4), torch.zeros(2, 4), nn.Identity(), "nope")


# ------------------------------------------------------------- 두 뷰 SpecAugment
@needs_golden
def test_two_view_specaugment_compounds_like_reference():
    """
    원본 train() 은 in-place SpecAugment 를 같은 배치에 2회 호출한다 — 두 번째
    뷰는 첫 뷰의 마스크 위에 누적된다. 우리는 view2 = specaug(view1) 로 재현한다.

    덤프와 같은 순서로 torch 스트림을 소모해야 한다(입력 randn 생성 포함) —
    gate(p >= randn)가 같은 위치의 난수를 봐야 하기 때문.
    """
    tw = torch.load(GOLDEN / "twoview_golden.pt")
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    x = torch.randn(2, 1, 798, 128)  # [B, 1, time, freq]
    assert torch.equal(x, torch.load(GOLDEN / "twoview_input.pt"))

    sa = SpecAugment(policy="icbhi_ast_sup", mask="mean")
    v1 = apply_specaug_paper(sa, x.squeeze(1))
    v2 = apply_specaug_paper(sa, v1)
    assert torch.equal(v1.unsqueeze(1), tw["v1"])
    assert torch.equal(v2.unsqueeze(1), tw["v2"])
    assert not torch.equal(v1, v2)  # 누적 마스킹이 실제로 일어났는지


# ------------------------------------------------------------------------ GRL
def test_grl_identity_forward_negated_scaled_backward():
    x = torch.randn(3, 5, requires_grad=True)
    y = ReverseLayerF.apply(x, 0.7)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.7))


# ------------------------------------------------------ 방법 분기 (스텁 합성)
class _AddOne(nn.Module):
    def forward(self, x):
        return x + 1.0


def test_sgscl_training_losses_two_views_compound_and_compose():
    """view2 = specaug(view1) 인지, 총손실 = CE + alpha*MetaCL 인지 스텁으로 검증."""
    torch.manual_seed(0)
    x = torch.randn(4, 6, 5)  # (B, T, F)
    labels = torch.tensor([0, 1, 2, 3])
    meta = torch.tensor([0, 0, 1, 1])
    feats_fn = lambda v: v.mean(dim=1)  # (B, F)
    classifier = nn.Linear(5, 4)
    projector = nn.Linear(5, 5)
    metacl = MetaCL(temperature=0.06)
    ce = nn.CrossEntropyLoss()

    logits, loss = sgscl_training_losses(
        method="sgscl", paper_features=feats_fn, x_paper=x, specaug=_AddOne(),
        classifier=classifier, domain_classifier=projector, metacl=metacl,
        loss_func=ce, labels=labels, meta_labels=meta,
        alpha=1.0, grl_alpha=1.0, target_type="project1_project2block",
    )

    f1 = feats_fn(x + 1.0)          # view1 = specaug(x)
    f2 = feats_fn(x + 2.0)          # view2 = specaug(view1) — 누적
    expected = ce(classifier(f1), labels) + metacl(projector(f1), projector(f2).detach(), meta)
    assert torch.allclose(loss, expected)
    assert torch.allclose(logits, classifier(f1))


def test_dat_losses_use_grl_and_unweighted_ce():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 5)
    labels = torch.tensor([0, 1, 0, 1])
    meta = torch.tensor([0, 2, 1, 3])
    feats_fn = lambda v: v.mean(dim=1)
    classifier = nn.Linear(5, 2)
    domain_head = nn.Linear(5, 4)
    ce = nn.CrossEntropyLoss()

    _, loss = sgscl_training_losses(
        method="dat", paper_features=feats_fn, x_paper=x, specaug=None,
        classifier=classifier, domain_classifier=domain_head, metacl=None,
        loss_func=ce, labels=labels, meta_labels=meta,
        alpha=1.0, grl_alpha=0.5, target_type="project1_project2block",
    )
    loss.backward()
    # GRL: 도메인 손실의 gradient 는 특징 쪽으로 부호가 뒤집혀 흐른다 —
    # 도메인 헤드 자체는 정방향 gradient 를 받는다 (0 이 아니어야 함)
    assert domain_head.weight.grad is not None and domain_head.weight.grad.abs().sum() > 0

    f1 = feats_fn(x)
    expected = ce(classifier(f1), labels) + 1.0 * nn.functional.cross_entropy(domain_head(f1), meta)
    assert torch.allclose(loss.detach(), expected.detach())


def test_dat_and_sgscl_require_meta_labels():
    with pytest.raises(ValueError, match="meta_labels"):
        sgscl_training_losses(
            method="sgscl", paper_features=lambda v: v.mean(dim=1),
            x_paper=torch.randn(2, 3, 4), specaug=None,
            classifier=nn.Linear(4, 2), domain_classifier=nn.Linear(4, 4),
            metacl=MetaCL(), loss_func=nn.CrossEntropyLoss(),
            labels=torch.tensor([0, 1]), meta_labels=None,
            alpha=1.0, grl_alpha=1.0, target_type="project1_project2block",
        )


# ----------------------------------------------------------------- AST (무거움)
@needs_golden
@needs_ast
def test_sgscl_ast_state_dict_matches_reference():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_sgscl")(n_cls=4, method="sgscl")
    ref = json.load(open(GOLDEN / "ast_state_dict_shapes.json"))
    ours = {k: list(v.shape) for k, v in m.state_dict().items()}

    def strip_head(d):
        # DeiT 골격의 미사용 분류 헤드(head/head_dist)는 timm 버전에 따라 있고 없음
        return {k: v for k, v in d.items() if not k.startswith(("v.head", "head"))}

    ref_s = strip_head(ref)
    core = strip_head({k: v for k, v in ours.items()
                       if k.startswith(("v.", "mlp_head.", "domain_mlp_head."))})
    assert not [k for k in ref_s if k not in core], "참조에 있는 키 누락"
    assert not {k for k in ref_s if ref_s[k] != core.get(k)}, "모양 불일치"
    assert not [k for k in core if k not in ref_s], "참조에 없는 키"
    # set_model 상당의 추가 모듈만 허용: classifier(=mlp_head 복사), domain_classifier
    others = {k.split(".")[0] for k in ours if k not in core and not k.startswith(("v.head", "head"))}
    assert others == {"classifier", "domain_classifier"}
    # 원본 quirk: SG-SCL 에서 domain_mlp_head 는 기본 527 차원인 채 미사용 잔존
    assert ours["domain_mlp_head.1.weight"] == [527, 768]


@needs_golden
@needs_ast
def test_sgscl_ast_forward_features_match_reference():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_sgscl")(n_cls=4, method="sgscl")
    m.eval()
    x = torch.load(GOLDEN / "features_input.pt")  # (2, 1, 798, 128)
    with torch.no_grad():
        feats = m.forward_features(x.squeeze(1))
    ref = torch.load(GOLDEN / "ast_features.pt")
    assert torch.allclose(feats, ref, atol=1e-4), (
        f"max|diff|={float((feats - ref).abs().max()):.3e}"
    )


@needs_ast
def test_sgscl_ast_training_forward_smoke_and_grad_flow():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_sgscl")(n_cls=4, method="sgscl")
    m.train()
    x = torch.randn(2, 128, 798)  # 툴킷 레이아웃 (freq_time 은 config 로; 기본 time_freq)
    m.input_layout = "freq_time"
    labels = torch.tensor([0, 1])
    meta = torch.tensor([0, 3])
    logits, loss = m.training_forward(x, labels, nn.CrossEntropyLoss(), meta_labels=meta)
    assert logits.shape == (2, 4) and torch.isfinite(loss)
    loss.backward()
    assert m.domain_classifier.linear1.weight.grad is not None  # MetaCL 경로로 gradient
    assert m.mlp_head[1].weight.grad is None  # 원본처럼 mlp_head 는 미사용(gradient 없음)

    m.eval()
    logits_eval, loss_eval = m.training_forward(x, labels, nn.CrossEntropyLoss(), meta_labels=meta)
    assert logits_eval.shape == (2, 4) and torch.isfinite(loss_eval)


# --------------------------------------------------------- SFT 래퍼 방법 확장
def test_sft_wrapper_sgscl_method_smoke():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("resnet18_sft")(
        n_cls=4, from_sl_official=False, method="sgscl",
        specaug_policy="icbhi_ast_sup",
    )
    m.train()
    x = torch.randn(2, 128, 798)
    logits, loss = m.training_forward(
        x, torch.tensor([0, 1]), nn.CrossEntropyLoss(), meta_labels=torch.tensor([1, 1])
    )
    assert logits.shape == (2, 4) and torch.isfinite(loss)
    loss.backward()
    assert isinstance(m.domain_classifier, Projector)
    assert m.domain_classifier.linear1.in_features == 512  # 비-AST: Projector(feat_dim, 768)
    assert m.domain_classifier.linear2.out_features == 768


def test_sft_wrapper_dat_method_smoke():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("resnet18_sft")(
        n_cls=4, from_sl_official=False, method="dat", m_cls=4,
    )
    m.train()
    x = torch.randn(2, 128, 798)
    logits, loss = m.training_forward(
        x, torch.tensor([0, 1]), nn.CrossEntropyLoss(), meta_labels=torch.tensor([0, 2])
    )
    assert torch.isfinite(loss)
    # 비-AST DAT: nn.Linear(feat_dim, m_cls) (원본 main.py:244)
    assert isinstance(m.domain_classifier, nn.Linear)
    assert tuple(m.domain_classifier.weight.shape) == (4, 512)


def test_sft_wrapper_ce_method_unchanged():
    """method 기본값(ce)·specaug 없음이면 기존 SFT 경로와 동일해야 한다."""
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("resnet18_sft")(n_cls=4, from_sl_official=False)
    m.train()
    x = torch.randn(2, 128, 798)
    torch.manual_seed(5)
    logits, loss = m.training_forward(x, torch.tensor([0, 1]), nn.CrossEntropyLoss())
    assert torch.allclose(logits, m(x))
    assert m.domain_classifier is None and m.specaug is None


# ------------------------------------------------------------- meta 라벨 전달
def test_build_meta_label_fn_dev_mapping():
    from rsc_toolkit.datasets.meta_labels import build_meta_label_fn

    fn = build_meta_label_fn("dev")
    assert fn({"meta": {"device": "Meditron"}}) == 0
    assert fn({"meta": {"device": "LittC2SE"}}) == 1
    assert fn({"meta": {"device": "Litt3200"}}) == 2
    assert fn({"meta": {"device": "AKGC417L"}}) == 3
    with pytest.raises(KeyError, match="device"):
        fn({"meta": {"device": "Unknown"}, "sample_id": "x"})
    with pytest.raises(NotImplementedError):
        build_meta_label_fn("loc")


def test_tuple_collate_passes_meta_labels_through():
    from rsc_toolkit.trainer.data.index_datamodule import _tuple_collate

    b2 = [(torch.zeros(3), 1, "a"), (torch.ones(3), 2, "b")]
    out = _tuple_collate(b2)
    assert len(out) == 2

    b3 = [(torch.zeros(3), 1, "a", 0), (torch.ones(3), 2, "b", 3)]
    x, y, meta = _tuple_collate(b3)
    assert torch.equal(meta, torch.tensor([0, 3]))
    assert torch.equal(y, torch.tensor([1, 2]))


def test_index_feature_dataset_appends_meta_label():
    from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset

    class _FakeIndexDS:
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {"audio": torch.zeros(1, 8), "label": 2, "sample_id": "s0",
                    "meta": {"device": "Litt3200"}}

    class _FakeFP:
        def make_features_cached(self, audio, sample_id):
            return torch.zeros(4)

    from rsc_toolkit.datasets.meta_labels import build_meta_label_fn

    ds = IndexFeatureDataset(_FakeIndexDS(), _FakeFP(), meta_label_fn=build_meta_label_fn("dev"))
    feats, label, sid, meta = ds[0]
    assert (label, sid, meta) == (2, "s0", 2)

    ds_plain = IndexFeatureDataset(_FakeIndexDS(), _FakeFP())
    assert len(ds_plain[0]) == 3
