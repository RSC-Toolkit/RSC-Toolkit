"""
BTS 묶음(Audio-CLAP·BTS)의 P0 테스트.

골든 출처:
- tests/golden/bts/meta_descriptions.json — 공식 util/icbhi_util.get_meta_text_descriptions 를
  실제 test.pt meta 28 조합 x 15 meta_mode 에 돌린 문장 (전수 206,940 건 대조 불일치 0 확인).
- tests/golden/bts/checkpoint_eval_record.json — 저자 best.pth 로드 평가 기록
  (AMP: Sp 82.58 / Se 45.11 / Score 63.85 = README). 1.8 GB 파일이라 여기서는 안 돌리고
  scripts/reproduction/eval_bts_checkpoint.py 로 재현한다.
- tests/golden/bts/pipeline_parity_test_split.json — 툴킷 48 kHz 파이프라인 vs 저자 .pt 대조 기록.
"""
import json
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.datasets.icbhi.meta_description import (
    META_MODES,
    describe_meta7,
    generate_meta_description,
    meta7_from_index_meta,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "bts"
CKPT = REPO / "pretrained_models" / "upstream" / "bts" / "best.pth"


def _hf_clap_cached() -> bool:
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    return any(p.name.startswith("models--laion--clap-htsat-unfused") for p in hub.glob("models--*"))


needs_clap = pytest.mark.skipif(not _hf_clap_cached(), reason="laion/clap-htsat-unfused 가 HF 캐시에 없음")


# ------------------------------------------------------------- 메타 문장
@pytest.mark.skipif(not (GOLDEN / "meta_descriptions.json").exists(), reason="골든 없음")
def test_meta_descriptions_match_reference_golden():
    golden = json.load(open(GOLDEN / "meta_descriptions.json"))
    assert len(golden) >= 15 * 20
    modes_seen = set()
    for g in golden:
        assert describe_meta7(g["meta7"], g["mode"]) == g["text"], (g["mode"], g["meta7"])
        modes_seen.add(g["mode"])
    assert modes_seen == set(META_MODES)


def test_generate_meta_description_examples():
    """원본 문장 형식 (논문 Fig. 2 예시와 동일 패턴)."""
    assert generate_meta_description("adult", "male", "trachea", "Meditron") == \
        "This sound was recorded from the trachea of an adult male patient, using a Meditron stethoscope."
    assert generate_meta_description("pediatric", "female", None, "AKGC417L") == \
        "This sound was recorded from a female pediatric patient, using an AKGC417L microphone."
    assert generate_meta_description("adult", None, None, None) == "This patient is an adult patient."
    assert generate_meta_description(None, None, "left anterior chest", None) == \
        "This sound was recorded from the left anterior chest."


def test_unknown_age_is_pediatric_like_reference():
    """원본 quirk: 결측 나이(-1)는 19 미만이라 pediatric 이 된다."""
    assert "pediatric" in describe_meta7([-1, 0, -1, -1, -1, 0, 0], "age")


def test_meta7_from_index_meta_matches_reference_layout():
    meta = {"age": 70.0, "sex": "F", "adult_BMI": 33.0, "child_weight": float("nan"),
            "child_height": None, "chest_location": "Ar", "device": "LittC2SE"}
    assert meta7_from_index_meta(meta) == [70.0, 1.0, 33.0, -1.0, -1.0, 2.0, 1.0]
    assert describe_meta7(meta7_from_index_meta(meta), "all") == \
        "This sound was recorded from the right anterior chest of an adult female patient, using a LittC2SE stethoscope."


# ---------------------------------------------------------- 데이터 로더
def test_precomputed_dataset_and_collate(tmp_path):
    from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataset, bts_collate

    items = [(torch.randn(1, 1001, 64), 2, torch.tensor([70., 1., 33., -1., -1., 2., 1.], dtype=torch.float64)),
             (torch.randn(1, 1001, 64), 0, torch.tensor([5., 0., -1., 20., 110., 0., 3.], dtype=torch.float64))]
    p = tmp_path / "fake.pt"
    torch.save(items, p)
    ds = BTSPrecomputedDataset(p)
    x, y, meta = bts_collate([ds[0], ds[1]])
    assert x.shape == (2, 1, 1001, 64) and y.tolist() == [2, 0]
    assert meta.shape == (2, 7) and meta.dtype == torch.float32


# ----------------------------------------------------------------- 모델
@needs_clap
def test_bts_model_state_dict_matches_reference_checkpoint_layout():
    from rsc_toolkit.models.bts import BTSModel

    m = BTSModel(n_cls=4)
    keys = set(m.state_dict())
    assert {"classifier.weight", "classifier.bias"} <= keys
    clap_keys = {k for k in keys if k.startswith("audio_features.")}
    assert len(clap_keys) == 447  # transformers 5.15.1 ClapModel == 저자 ckpt 키 수
    assert tuple(m.classifier.weight.shape) == (4, 1024)  # concat: text 512 + audio 512

    if CKPT.exists():
        m.load_reference_checkpoint(CKPT)  # strict=True — 키 집합이 정확히 같아야 통과


@needs_clap
def test_bts_training_forward_uses_text_and_audio():
    from rsc_toolkit.models.bts import BTSModel

    m = BTSModel(n_cls=4).eval()
    x = torch.randn(2, 1, 1001, 64)
    meta = torch.tensor([[70., 1., 33., -1., -1., 2., 1.], [5., 0., -1., 20., 110., 0., 3.]])
    with torch.no_grad():
        logits, loss = m.training_forward(x.squeeze(1), torch.tensor([0, 1]), nn.CrossEntropyLoss(), meta_labels=meta)
        same_audio_other_meta = m(x, meta[[1, 0]])
    assert logits.shape == (2, 4) and torch.isfinite(loss)
    assert not torch.allclose(logits, same_audio_other_meta)  # 메타 문장이 출력에 실제로 반영

    with pytest.raises(ValueError, match="meta"):
        m.training_forward(x, torch.tensor([0, 1]), nn.CrossEntropyLoss())


@needs_clap
def test_audioclap_accepts_3d_and_4d_input():
    from rsc_toolkit.models.factory import ModelFactory

    m = ModelFactory.get_model_class("audioclap")(n_cls=4).eval()
    x = torch.randn(2, 1, 1001, 64)
    with torch.no_grad():
        a = m(x)
        b = m(x.squeeze(1))
        logits, loss = m.training_forward(x.squeeze(1), torch.tensor([0, 3]), nn.CrossEntropyLoss(),
                                          meta_labels=torch.zeros(2, 7))
    assert torch.allclose(a, b) and logits.shape == (2, 4) and torch.isfinite(loss)
