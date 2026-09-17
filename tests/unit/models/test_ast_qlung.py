"""
QLung(Interspeech 2026, RSC-Toolkit/QLung 46408dd) P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_qlung_goldens.py
- aqs     : 공식 util/get_acoustic_features.py (librosa 0.11) — ICBHI train 사이클 6개 + 합성 2개의 rms/엔트로피/AQS 3변형
- specaug : 공식 util/augmentation.py SpecAugment(icbhi_ast_sup) 6회 호출 (게이트 통과 5/6)
- loss    : 공식 method/loss.py AngularClassifier / AngularMarginLoss — 참조 venv(timm 0.4.5, CPU 패치)
- model   : 공식 models/ast.py(798/128 quirk, AudioSet ckpt) forward + det_fill 헤드 → 로짓·총손실
"""
import json
import pathlib
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsc_toolkit.methods.qlung import (
    ICBHI_TRAIN_CLASS_NUMS,
    AngularClassifier,
    AngularMarginLoss,
    aqs_from_features,
    class_imbalance_margins,
    compute_acoustic_features,
    qlung_dual_loss,
    quality_score,
    rms_energy,
    spectral_entropy,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "qlung"
AUDIOSET = any((REPO / c).exists() for c in (
    "pretrained_models/upstream/audioset_0.4593.pth", "pretrained_models/audioset_10_10_0.4593.pth"))
needs_golden = pytest.mark.skipif(not (GOLDEN / "loss_golden.pt").exists(),
                                  reason="골든 없음 — scripts/reproduction/dump_qlung_goldens.py")
needs_ast = pytest.mark.skipif(not AUDIOSET, reason="AudioSet AST ckpt 없음")
needs_icbhi = pytest.mark.skipif(not (REPO / ".datasets/icbhi/index/index_cycle.csv").exists(), reason="ICBHI 인덱스 없음")
needs_sprs = pytest.mark.skipif(not (REPO / ".datasets/sprsound/index/index_cycle.csv").exists(), reason="SPRSound 인덱스 없음")


def _load(name):
    return torch.load(GOLDEN / name, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ AQS
@needs_golden
def test_aqs_three_variants_match_librosa_reference():
    g = _load("aqs_golden.pt")
    for row in g["rows"]:
        f = compute_acoustic_features(row["wave"])
        assert abs(f["rms"] - row["rms"]) < 1e-6, row["sample_id"]
        assert abs(f["entropy"] - row["entropy"]) < 1e-5, row["sample_id"]
        # 코드 그대로 (rms, rms) — 기본값
        assert abs(aqs_from_features(f["rms"], f["entropy"]) - row["aqs_code_rms"]) < 1e-6
        # 코드식 + 엔트로피
        assert abs(aqs_from_features(f["rms"], f["entropy"], entropy_source="entropy") - row["aqs_code_entropy"]) < 1e-5
        # 논문 식 (5)
        assert abs(aqs_from_features(f["rms"], f["entropy"], entropy_source="entropy", formula="paper") - row["aqs_paper"]) < 1e-5
        # 유한 프레임 사이클에서는 nanmean == 문자 그대로
        assert abs(f["entropy_nanmean"] - f["entropy"]) < 1e-6
        assert abs(aqs_from_features(f["rms"], f["entropy"], entropy_source="entropy_nanmean", entropy_nanmean=f["entropy_nanmean"])
                   - row["aqs_code_entropy"]) < 1e-5


def test_spectral_entropy_nan_policy_on_digital_silence():
    y = torch.cat([torch.zeros(3000), torch.randn(125000)])   # 앞 프레임 Σ|S| = 0 → 원본은 nan
    assert torch.isnan(spectral_entropy(y))
    nm = spectral_entropy(y, nan_policy="nanmean")
    assert torch.isfinite(nm) and 0.0 < float(nm) <= 1.0
    f = compute_acoustic_features(y)
    assert np.isnan(f["entropy"]) and np.isfinite(f["entropy_nanmean"])
    assert np.isnan(aqs_from_features(f["rms"], f["entropy"], entropy_source="entropy"))
    assert 0.0 <= aqs_from_features(f["rms"], f["entropy"], entropy_source="entropy_nanmean", entropy_nanmean=f["entropy_nanmean"]) <= 1.0


def test_aqs_code_formula_is_paper_formula_shifted_and_clipped():
    # 코드식 = (−αH + βR + α)/(α+β); α+β=1 이면 논문식 − (1−α) … 즉 상수 0.3 차이
    e, r = 0.4, 0.3
    code = float(quality_score(e, r, formula="code"))
    paper = float(quality_score(e, r, formula="paper"))
    assert abs((paper - code) - 0.3) < 1e-12
    assert float(quality_score(0.0, 5.0, formula="paper")) == 1.0  # clip


def test_rms_entropy_frame_counts_follow_librosa_center_padding():
    y = torch.randn(128000)
    # 1 + len//hop 프레임 (center=True) — 프레임 수가 다르면 평균이 달라진다
    pad = 512
    frames = F.pad(y, (pad, pad)).unfold(0, 1024, 512)
    assert frames.shape[0] == 251
    assert torch.isfinite(rms_energy(y)) and torch.isfinite(spectral_entropy(y))
    assert 0.0 < float(spectral_entropy(y)) <= 1.0


# ------------------------------------------------------------------ 마진 / DFAM
def test_class_imbalance_margins_from_icbhi_prior():
    m = class_imbalance_margins(ICBHI_TRAIN_CLASS_NUMS, m_target=0.2)
    n = torch.tensor(ICBHI_TRAIN_CLASS_NUMS, dtype=torch.float64)
    expected = -torch.log(n / n.sum()) * (0.2 / np.log(4))
    assert m.dtype == torch.float64 and torch.allclose(m, expected)
    assert torch.all(m[1:] > m[:-1])  # 희소 클래스일수록 큰 마진


@needs_golden
def test_angular_and_dfam_logits_and_losses_match_reference():
    g = _load("loss_golden.pt")
    clf = AngularClassifier(768, 4, s=g["s_a"])
    dfam = AngularMarginLoss(768, 4, s=g["s_d"], m=g["m"], gamma=g["gamma"], m_target=g["m_target"],
                             class_nums=g["class_nums"], easy_margin=False)
    with torch.no_grad():
        clf.weight.copy_(g["clf_weight"])
        dfam.weight.copy_(g["dfam_weight"])
    assert torch.allclose(dfam.class_margin, g["class_margin"], atol=1e-12)

    logits, total, class_loss, aqm_loss = qlung_dual_loss(g["features"], g["labels"], g["aqs"], clf, dfam,
                                                          nn.CrossEntropyLoss(), lambda_aqm=g["lambda"])
    assert logits.dtype == torch.float32 and torch.allclose(logits, g["angular_logits"], atol=1e-5)
    dfam_logits = dfam(F.normalize(g["features"]), g["labels"], g["aqs"])
    assert dfam_logits.dtype == torch.float64  # float64 승격 quirk
    assert torch.allclose(dfam_logits, g["dfam_logits"], atol=1e-6)
    assert abs(float(class_loss) - float(g["class_loss"])) < 1e-6
    assert abs(float(aqm_loss) - float(g["aqm_loss"])) < 1e-6
    assert abs(float(total) - float(g["total_loss"])) < 1e-6
    assert total.dtype == torch.float64


@needs_golden
def test_easy_margin_branches_match_reference():
    g = _load("loss_golden.pt")
    fn = F.normalize(g["features"])
    hard = AngularMarginLoss(768, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=g["class_nums"], easy_margin=False)
    easy = AngularMarginLoss(768, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=g["class_nums"], easy_margin=True)
    with torch.no_grad():
        hard.weight.copy_(g["dfam_weight"])
        easy.weight.copy_(g["dfam_weight"])
    out_h = hard(fn, g["labels"], g["aqs"])
    out_e = easy(fn, g["labels"], g["aqs"])
    assert torch.allclose(out_e, g["dfam_logits_easy"], atol=1e-6)
    # 골든의 행 6·7 은 cos≈−1 (cos−th ≤ 0) → hard: cos−mm, easy: cos 그대로 → 서로 달라야 한다
    y = g["labels"]
    for i in (6, 7):
        assert not torch.isclose(out_h[i, y[i]], out_e[i, y[i]])
        m = hard.margins(y, g["aqs"])[i]
        cos = g["dfam_cosine"][i, y[i]].double()
        assert torch.isclose(out_h[i, y[i]], (cos - torch.sin(torch.pi - m) * m) * 15, atol=1e-6)
        assert torch.isclose(out_e[i, y[i]], cos * 15, atol=1e-6)
    for i in range(6):  # 일반 행: phi = cos(θ+m)
        m = hard.margins(y, g["aqs"])[i]
        cos = g["dfam_cosine"][i, y[i]].double()
        phi = cos * torch.cos(m) - torch.sqrt((1 - cos * cos).clamp(0, 1)) * torch.sin(m)
        assert torch.isclose(out_h[i, y[i]], phi * 15, atol=1e-6)


def test_dfam_margin_composition_gamma_kappa():
    dfam = AngularMarginLoss(8, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=ICBHI_TRAIN_CLASS_NUMS)
    y = torch.tensor([0, 3])
    q = torch.tensor([1.0, 0.0])
    m = dfam.margins(y, q)
    assert torch.isclose(m[0], 0.5 * 0.5 * 1.0 + 0.5 * dfam.class_margin[0])
    assert torch.isclose(m[1], 0.5 * dfam.class_margin[3])


# ------------------------------------------------------------------ SpecAugment 게이트
@needs_golden
@pytest.mark.exact_golden  # 비트 단위 비교 — 개발 머신 전용 (CPU/BLAS/스레드 수가 다르면 마지막 자리에서 어긋난다)
def test_specaug_batch_gate_and_masks_match_reference():
    from rsc_toolkit.augmentations.specaugment import SpecAugment
    from rsc_toolkit.methods.sgscl import apply_specaug_paper

    g = _load("specaug_golden.pt")
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    x = torch.randn(4, 1, 798, 128)
    assert torch.equal(x, _load("specaug_input.pt"))
    sa = SpecAugment(policy="icbhi_ast_sup", mask="mean")
    applied = []
    for ref in g["outs"]:
        out = apply_specaug_paper(sa, x.squeeze(1).clone()).unsqueeze(1)
        assert torch.equal(out, ref)
        applied.append(not torch.equal(out, x))
    assert True in applied and False in applied  # randn 게이트가 실제로 갈렸는지 (골든: 5/6 적용)


# ------------------------------------------------------------------ 모델 (무거움)
@needs_golden
@needs_ast
def test_qlung_ast_full_forward_and_loss_match_reference():
    from rsc_toolkit.models.factory import ModelFactory

    g = _load("model_golden.pt")
    m = ModelFactory.get_model_class("ast_qlung")(n_cls=4, method="aqm")
    with torch.no_grad():
        m.classifier.weight.copy_(g["clf_weight"])
        m.dfam.weight.copy_(g["dfam_weight"])
    m.eval()
    x = g["input"].squeeze(1)  # (2, 798, 128) time_freq
    with torch.no_grad():
        feats = m.forward_features(x)
    assert torch.allclose(feats, g["features"], atol=1e-4), float((feats - g["features"]).abs().max())
    # 참조 특징으로 헤드·손실 대조 (백본 오차와 분리)
    logits, total, cl, al = qlung_dual_loss(g["features"], g["labels"], g["aqs"], m.classifier, m.dfam,
                                            nn.CrossEntropyLoss(), lambda_aqm=0.4)
    assert torch.allclose(logits, g["angular_logits"], atol=1e-5)
    assert abs(float(total) - float(g["total_loss"])) < 1e-6
    # 평가 경로: 정규화 특징 → AngularClassifier, loss = CE(logits.float64)
    with torch.no_grad():
        ev_logits, ev_loss = m.training_forward(x, g["labels"], nn.CrossEntropyLoss(), meta_labels=g["aqs"])
    assert torch.allclose(ev_logits, logits, atol=1e-3)
    assert ev_loss.dtype == torch.float64
    # 우리 백본 특징으로 학습 경로(SpecAug 게이트 강제 통과) 총손실도 골든 근방
    m.train()
    torch.manual_seed(0)
    tr_logits, tr_loss = m.training_forward(x, g["labels"], nn.CrossEntropyLoss(), meta_labels=g["aqs"])
    assert tr_logits.shape == (2, 4) and torch.isfinite(tr_loss)
    tr_loss.backward()
    assert m.dfam.weight.grad is not None and m.classifier.weight.grad is not None
    assert m.mlp_head[1].weight.grad is None  # 미사용 mlp_head 잔존


@needs_golden
@needs_ast
def test_qlung_ast_state_dict_matches_reference_plus_heads():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("ast_qlung")(n_cls=4, method="aqm")
    ref = json.load(open(GOLDEN / "ast_state_dict_shapes.json"))
    ours = {k: list(v.shape) for k, v in m.state_dict().items()}

    def strip(d):
        return {k: v for k, v in d.items() if not k.startswith(("v.head", "head"))}

    ref_s = strip(ref)
    core = strip({k: v for k, v in ours.items() if k.startswith(("v.", "mlp_head."))})
    # 참조(sgscl 클론 ast.py)는 미사용 domain_mlp_head(527) 를 갖는다 — QLung 경로에서 쓰이지 않아 재현하지 않는다
    ref_core = {k: v for k, v in ref_s.items() if not k.startswith("domain_mlp_head.")}
    assert not [k for k in ref_core if k not in core], "참조에 있는 키 누락"
    assert not {k for k in ref_core if ref_core[k] != core.get(k)}, "모양 불일치"
    assert not [k for k in core if k not in ref_core], "참조에 없는 키"
    others = {k.split(".")[0] for k in ours if k not in core and not k.startswith(("v.head", "head"))}
    assert others == {"classifier", "dfam"}
    assert ours["classifier.weight"] == [4, 768] and ours["dfam.weight"] == [4, 768]
    assert "dfam.class_margin" not in ours  # 상수 buffer 는 state_dict 밖 (원본도 state_dict 밖)


# ------------------------------------------------------------------ method=ce quirk
def _tiny_qlung(method):
    """AST 없이 헤드 경로만 검증하기 위한 스텁: forward_features 를 평균 풀링으로 대체."""
    from rsc_toolkit.models.ast_qlung import QLungAST

    m = QLungAST.__new__(QLungAST)
    nn.Module.__init__(m)
    m.final_feat_dim = 6
    m.input_layout = "time_freq"
    m.method = method
    m.lambda_aqm = 0.4
    m.n_cls = 4
    m.mlp_head = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, 4))
    if method == "aqm":
        m.classifier = AngularClassifier(6, 4, s=37)
        m.dfam = AngularMarginLoss(6, 4, s=15, m=0.5, gamma=0.5, m_target=0.2, class_nums=ICBHI_TRAIN_CLASS_NUMS)
    else:
        from copy import deepcopy
        m.classifier = deepcopy(m.mlp_head)
        m.dfam = None
    from rsc_toolkit.augmentations.specaugment import SpecAugment
    m.specaug = SpecAugment(policy="none")
    m.forward_features = lambda x: x.mean(dim=1)
    return m


def test_method_ce_train_unnormalized_eval_normalized_quirk():
    torch.manual_seed(0)
    m = _tiny_qlung("ce")
    x = torch.randn(3, 5, 6) * 4
    y = torch.tensor([0, 1, 2])
    ce = nn.CrossEntropyLoss()
    m.train()
    logits_tr, loss_tr = m.training_forward(x, y, ce)
    feats = x.mean(dim=1)
    assert torch.allclose(logits_tr, m.classifier(feats))                 # 비정규화
    assert loss_tr.dtype == torch.float32
    m.eval()
    logits_ev, loss_ev = m.training_forward(x, y, ce)
    assert torch.allclose(logits_ev, m.classifier(F.normalize(feats)))    # 정규화
    assert loss_ev.dtype == torch.float64
    assert not torch.allclose(logits_tr, logits_ev)


def test_method_aqm_requires_aqs_and_uses_dual_loss():
    torch.manual_seed(0)
    m = _tiny_qlung("aqm")
    x = torch.randn(4, 5, 6)
    y = torch.tensor([0, 1, 2, 3])
    m.train()
    with pytest.raises(ValueError, match="AQS"):
        m.training_forward(x, y, nn.CrossEntropyLoss())
    aqs = torch.tensor([0.6, 0.5, 0.7, 0.4])
    logits, loss = m.training_forward(x, y, nn.CrossEntropyLoss(), meta_labels=aqs)
    ref_logits, ref_loss, _, _ = qlung_dual_loss(x.mean(dim=1), y, aqs, m.classifier, m.dfam, nn.CrossEntropyLoss(), 0.4)
    assert torch.allclose(logits, ref_logits) and torch.isclose(loss, ref_loss)


# ------------------------------------------------------------------ MA 제외 집합
def test_live_weight_averaging_excludes_dfam():
    from rsc_toolkit.trainer.callbacks.qlung_ma_update import LiveWeightAveragingExcluding

    class _PL:
        pass

    m = _tiny_qlung("aqm")
    pl = _PL()
    pl.rsc_model = m
    cb = LiveWeightAveragingExcluding(beta=0.6, exclude_prefixes=["dfam."])
    cb.on_train_batch_start(None, pl, None, 0)
    prev = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    cb.on_train_batch_end(None, pl, None, None, 0)
    cur = m.state_dict()
    for k in cur:
        if k.startswith("dfam."):
            assert torch.allclose(cur[k], prev[k] + 1.0)               # 제외: 현재 값 유지
        else:
            assert torch.allclose(cur[k], prev[k] * 0.6 + (prev[k] + 1.0) * 0.4)
    # 모델 속성 기본값도 dfam 제외
    from rsc_toolkit.models.ast_qlung import QLungAST
    assert QLungAST.ma_exclude_prefixes == ("dfam.",)
    assert LiveWeightAveragingExcluding(beta=0.6)._excluded(m) == ("dfam.",)


# ------------------------------------------------------------------ 데이터 경로
def test_qlung_collate_float_aqs():
    from rsc_toolkit.datasets.qlung_data import qlung_collate

    x, y, aqs = qlung_collate([(torch.zeros(2, 3), 1, "a", 0.61), (torch.ones(2, 3), 0, "b", 0.4)])
    assert aqs.dtype == torch.float64 and torch.allclose(aqs, torch.tensor([0.61, 0.4], dtype=torch.float64))  # 공식 default_collate(python float) 와 동일
    assert y.dtype == torch.long
    assert len(qlung_collate([(torch.zeros(2), 1, "a")])) == 2


@needs_icbhi
def test_aqs_cache_deterministic_and_variant_pure(tmp_path):
    from rsc_toolkit.datasets.qlung_data import AQSConfig, AcousticFeatureCache, QLungAQSIndexDataset, precompute_acoustic_features
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    pre = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat_fade"))
    kw = dict(index_csv=REPO / ".datasets/icbhi/index/index_cycle.csv", data_root=REPO / ".datasets/icbhi",
              sampling_rate=16000, sample_level="cycle", split="train",
              split_file=REPO / "rsc_toolkit/datasets/icbhi/data/official_split.txt", audio_preprocessor=pre)
    c1 = AcousticFeatureCache(tmp_path / "a.pt")
    ds = QLungAQSIndexDataset(**kw, aqs=AQSConfig(), acoustic_cache=c1)
    ds.df = ds.df.iloc[:6].reset_index(drop=True)
    precompute_acoustic_features(ds, c1, num_workers=0, batch_size=3)
    c2 = AcousticFeatureCache(tmp_path / "b.pt")
    ds2 = QLungAQSIndexDataset(**kw, aqs=AQSConfig(entropy_source="entropy"), acoustic_cache=c2)
    ds2.df = ds2.df.iloc[:6].reset_index(drop=True)
    precompute_acoustic_features(ds2, c2, num_workers=0, batch_size=2)
    assert c1.data == c2.data                                   # 결정성 (변형 무관 캐시)
    assert (tmp_path / "a.pt").exists() and torch.load(tmp_path / "a.pt", weights_only=False) == c1.data
    it, it2 = ds[0], ds2[0]
    assert it["aqs"] == aqs_from_features(it["rms"], it["entropy"])
    assert it2["aqs"] == aqs_from_features(it["rms"], it["entropy"], entropy_source="entropy")
    assert "entropy_nanmean" in c1.data[it["sample_id"]]
    assert it["aqs"] != it2["aqs"]
    assert 0.0 <= it["aqs"] <= 1.0


@needs_sprs
def test_sprsound_4class_mapping_counts_match_paper():
    from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4
    from rsc_toolkit.datasets.sprsound_ood_fbank import (PAPER_MERGED_COUNTS, TOOLKIT_4CLASS_COUNTS, build_sprsound_ood_datamodule,
                                                        class_counts_4)

    assert SPRSOUND_7_TO_4 == {0: 0, 1: 1, 3: 1, 2: 2, 4: 3}          # Rhonchi(5)·Stridor(6) 제외 (2026-09-16 사용자 규칙)
    dm = build_sprsound_ood_datamodule(["train", "test"], project_root=str(REPO), overrides={"qlung": {"precompute": False}})
    ds = dm._build_index_dataset(splits=["train", "test"])
    assert len(ds) == 8085 - 39 - 15 and tuple(class_counts_4(ds)) == TOOLKIT_4CLASS_COUNTS   # 6199/1044/757/31 = 8,031
    assert PAPER_MERGED_COUNTS == (6199, 1044, 811, 31)               # 논문 3.1 의 wheeze 병합 분포는 기록만 (툴킷 뷰와 다름)
    inter = dm._build_index_dataset(splits=["test"])
    assert len(inter) == 1429 and class_counts_4(inter) == [1040, 83, 305, 1]
