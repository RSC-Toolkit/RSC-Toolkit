"""
PC-MCL 라벨 공간 테스트 — ICBHI 는 **저자 동작과 같음을 고정**하고, SPRSound 는 **우리가 정한 규약을 고정**한다.

가장 중요한 것은 `test_icbhi_*` 다: 라벨 공간 추상화를 넣으면서 ICBHI 경로가 한 비트도 안 바뀌었음을 여기서 못 박는다.
(런 단위 대조는 `test_pcmcl.py` 의 골든 테스트가 맡는다.)
"""
import pathlib
import pytest
import torch

from rsc_toolkit.methods.pcmcl import multilabel_binary_to_label4
from rsc_toolkit.methods.pcmcl_labelspace import ICBHI4, SPRSOUND7, get_label_space

# `pcmcl_concat` 은 모듈 최상단에서 lightning 을 끌어온다(pcmcl_concat.py:523). 수집 단계에 그게 실리면
# 전역 상태가 평소보다 일찍 바뀌므로, test_pcmcl.py 와 같은 관례로 **함수 안에서** 부른다.


def _author_fns():
    from rsc_toolkit.datasets.pcmcl_concat import convert_to_multilabel, multilabel_mixing_combinations

    return convert_to_multilabel, multilabel_mixing_combinations




# ====================================================================== ICBHI 회귀 (저자 동작 고정)
def test_icbhi_encode_matches_author_function():
    """LabelSpace.to_multilabel == 원본 _convert_to_multilabel."""
    convert_to_multilabel, _ = _author_fns()
    for y in range(4):
        assert ICBHI4.to_multilabel(y) == convert_to_multilabel(y), y


def test_icbhi_decode_matches_author_conversion():
    """LabelSpace.decode_binary == conversion.py convert_original(additive), 8개 조합 전수."""
    combos = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    assert ICBHI4.decode_binary(combos).tolist() == multilabel_binary_to_label4(combos).tolist()
    assert ICBHI4.decode_binary(combos).tolist() == [0, 2, 1, 3, 0, 2, 1, 3]


def test_icbhi_buckets_match_author_rule():
    """포함(⊇) 규칙 == 원본의 '0·1·2 는 비트, 3 은 original_label==3'. 4개 클래스 전수."""
    convert_to_multilabel, _ = _author_fns()
    for y in range(4):
        ml = convert_to_multilabel(y)
        author = [k for k in range(3) if ml[k] > 0] + ([3] if y == 3 else [])
        assert ICBHI4.buckets_for(ml) == sorted(author), (y, ml)


def test_icbhi_combinations_unchanged():
    _, multilabel_mixing_combinations = _author_fns()
    for strategy in ("additive", "dominant"):
        assert ICBHI4.mixing_combinations(strategy) == multilabel_mixing_combinations(strategy)


def test_icbhi_pairs_are_union_of_attributes():
    """저자의 additive 목록이 '비순서 쌍 전부 + 합집합 라벨' 임을 실증한다 — SPRSound 확장의 근거."""
    combos = ICBHI4.mixing_combinations("additive")
    pairs = [c for c in combos if ICBHI4.sources_in(c) == 2]
    assert len(pairs) == 10                                    # C(4+1, 2)
    assert {(c[0], c[1]) if c[0] <= c[1] else (c[1], c[0]) for c in pairs} == \
           {(i, j) for i in range(4) for j in range(i, 4)}
    for c in pairs:
        union = [max(a, b) for a, b in zip(ICBHI4.to_multilabel(c[0]), ICBHI4.to_multilabel(c[1]))]
        assert ICBHI4.label_in(c) == union, c
    triples = [c for c in combos if ICBHI4.sources_in(c) == 3]
    assert len(triples) == 1 and (triples[0][0], triples[0][1], triples[0][2]) == (0, 1, 2)


def test_sources_in_reproduces_author_length_branches():
    """원본은 길이 6/7 로 소스 개수를 분기했다 — 되계산이 같은 답을 준다."""
    for c in ICBHI4.mixing_combinations("additive"):
        assert ICBHI4.sources_in(c) == (2 if len(c) == 6 else 3)


# ====================================================================== SPRSound (툴킷 규약 고정)
def test_sprsound_decomposition_is_lossless():
    """7 클래스가 서로 다른 속성 벡터를 갖고, 복호가 원 클래스를 되돌린다."""
    vecs = [tuple(SPRSOUND7.to_multilabel(y)) for y in range(7)]
    assert len(set(vecs)) == 7
    bits = torch.tensor(vecs)
    assert SPRSOUND7.decode_binary(bits).tolist() == list(range(7))


def test_sprsound_attribute_sharing_is_real():
    """MCL 이 사는 이유 — 하나의 속성이 여러 클래스에서 켜진다."""
    discont = [y for y in range(7) if SPRSOUND7.to_multilabel(y)[1] > 0]
    cont = [y for y in range(7) if SPRSOUND7.to_multilabel(y)[2] > 0]
    assert discont == [1, 3, 4]          # Fine · Coarse · Wheeze+Crackle
    assert cont == [2, 4, 5, 6]          # Wheeze · Wheeze+Crackle · Rhonchi · Stridor


def test_sprsound_axes_mirror_icbhi():
    """속성 0·1·2 의 역할이 ICBHI 와 같아야 PM 병리 프로필 코드가 그대로 돈다."""
    assert SPRSOUND7.attr_names[:3] == ("normal", "discont", "cont")
    assert SPRSOUND7.profile_attrs == ICBHI4.profile_attrs == (0, 1, 2)
    # Wheeze+Crackle 은 ICBHI 의 both 와 같은 자리(불연속 ∧ 연속)
    assert SPRSOUND7.to_multilabel(4)[1:3] == [1, 1]


def test_sprsound_buckets_pool_by_attribute():
    """버킷 1(Fine Crackle) 이 불연속 계열 전부를 모은다 — 저자 코드에서 both 가 crackle 버킷에 들어가던 것과 같은 구조."""
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(3)) == [1, 3]      # Coarse → 불연속·자기자신
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(4)) == [1, 2, 4]   # W+C → 불연속·연속·자기자신
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(5)) == [2, 5]      # Rhonchi → 연속·자기자신
    assert SPRSOUND7.buckets_for(SPRSOUND7.to_multilabel(0)) == [0]


def test_sprsound_combinations_are_all_pairs_with_union_labels():
    combos = SPRSOUND7.mixing_combinations("additive")
    assert len(combos) == 28                                   # C(7+1, 2)
    assert all(SPRSOUND7.sources_in(c) == 2 for c in combos)   # 3-소스 조합은 없다(모듈 docstring 참조)
    assert {(c[0], c[1]) for c in combos} == {(i, j) for i in range(7) for j in range(i, 7)}
    for c in combos:
        union = [max(a, b) for a, b in zip(SPRSOUND7.to_multilabel(c[0]), SPRSOUND7.to_multilabel(c[1]))]
        assert SPRSOUND7.label_in(c) == union, c


def test_sprsound_decode_priority_is_documented_behaviour():
    """존재하지 않는 속성 조합의 복호 — **우리가 정한 규칙**이라 여기서 고정한다."""
    ls = SPRSOUND7
    # 정상 비트가 켜져 있어도 이상음이 이긴다 (ICBHI 와 같은 구조)
    assert ls.decode_binary(torch.tensor([[1, 1, 0, 0, 0, 0]])).tolist() == [1]
    # 아무 속성도 안 켜지면 정상
    assert ls.decode_binary(torch.tensor([[0, 0, 0, 0, 0, 0]])).tolist() == [0]
    # 불연속+연속이 다른 무엇보다 먼저 — coarse 가 같이 켜져도 Wheeze+Crackle
    assert ls.decode_binary(torch.tensor([[0, 1, 1, 1, 0, 0]])).tolist() == [4]
    # coarse 는 불연속을 수식할 뿐 — 불연속 없이 coarse 만 켜지면 무시된다
    assert ls.decode_binary(torch.tensor([[0, 0, 1, 1, 0, 0]])).tolist() == [2]
    # rhonchi 와 stridor 가 동시에 켜지면 rhonchi 가 이긴다 (자의적 — 논문 각주 대상)
    assert ls.decode_binary(torch.tensor([[0, 0, 1, 0, 1, 1]])).tolist() == [5]


def test_sprsound_rejects_dominant_strategy():
    with pytest.raises(NotImplementedError):
        SPRSOUND7.mixing_combinations("dominant")


def test_out_of_range_label_raises_instead_of_corrupting():
    """가장 중요한 가드 — 4-class 변환기에 7-class 를 넣으면 라벨 4·5·6 이 조용히 [0,0,0] 이 됐었다."""
    convert_to_multilabel, _ = _author_fns()
    assert convert_to_multilabel(5) == [0, 0, 0]        # 옛 동작(참고): 조용히 소실
    with pytest.raises(ValueError, match="클래스 인덱스"):
        ICBHI4.to_multilabel(5)                         # 새 동작: 즉시 실패


def test_get_label_space_rejects_unknown():
    with pytest.raises(ValueError, match="label_space"):
        get_label_space("icbhi7")


# ====================================================================== 모델 배선
def test_model_requires_matching_n_cls():
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    with pytest.raises(ValueError, match="속성 수"):
        BEATsPCMCL(n_cls=3, label_space="sprsound7")
    with pytest.raises(ValueError, match="속성 수"):
        BEATsPCMCL(n_cls=7, label_space="sprsound7")     # 7 은 클래스 수이지 속성 수가 아니다


@pytest.mark.skipif(not pathlib.Path("pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt").exists(), reason="BEATs 체크포인트 없음")  # needs_beats_ckpt
def test_pm_head_survives_in_sprsound_multilabel():
    """C 안의 핵심 — 다중라벨 경로를 유지하므로 PM(환자 일관) 헤드가 그대로 살아 있다."""
    from rsc_toolkit.models.beats_pcmcl import BEATsPCMCL

    m = BEATsPCMCL(n_cls=6, label_space="sprsound7", enable_ssl=True)
    assert m.pm_head is not None and m.pm_head.ssl_classifier.out_features == 2
    assert m.classifier.out_features == 6
    # 검증 경로: 7-class long → (B, 7) one-hot + 6-속성 BCE
    onehot = m.ls.logits_to_onehot(torch.randn(5, 6))
    assert onehot.shape == (5, 7) and set(onehot.sum(1).tolist()) == {1.0}
