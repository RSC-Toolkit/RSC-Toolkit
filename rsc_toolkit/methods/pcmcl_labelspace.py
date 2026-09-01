"""
PC-MCL 의 **라벨 공간** — 클래스 인덱스 ↔ 속성 벡터 변환, 연결 조합, 버킷 규칙을 한곳에 모은다.

## 왜 필요한가

PC-MCL(arXiv 2601.17080) 의 MCL 은 "클래스가 몇 개냐" 가 아니라 **"라벨을 속성으로 쪼갤 수 있느냐"** 에 걸려 있다.
ICBHI 4-class 가 마침 2개의 이진 속성(crackle 유무 × wheeze 유무)이라, 저자 코드는 4-way 가 아니라
3-label `(normal, crackle, wheeze)` 로 학습한다 — `both` 가 `[0,1,1]` 이 되면서 crackle 검출기가
class 1 과 class 3 양쪽에서 학습되는 **속성 공유**가 MCL 의 실체다.

그래서 데이터셋을 옮길 때 필요한 것은 헤드 크기 변경이 아니라 **그 데이터셋의 속성 분해**다.
이 모듈이 그 분해를 담는다.

## 두 라벨 공간

| | ICBHI4 (저자 정의) | SPRSOUND7 (툴킷 정의 — 확장) |
|---|---|---|
| 클래스 | 4 | 7 |
| 속성 | 3 `normal · crackle · wheeze` | 6 `normal · discont · cont · coarse · rhonchi · stridor` |
| 출처 | `util/dataset_pcmcl.py _convert_to_multilabel` | **우리가 정의했다** — 논문·저자 코드에 없다 |

`discont`(불연속 부잡음) 는 ICBHI 의 `crackle` 축을, `cont`(연속 부잡음) 는 `wheeze` 축을 그대로 잇는다.
그래서 속성 0·1·2 의 역할이 두 공간에서 같고, PM(환자 매칭) 의 병리 프로필 분류가 코드 변경 없이 성립한다.

## 무엇이 저자 것이고 무엇이 우리 것인가 (논문에 반드시 명시)

- **저자 것**: ICBHI 분해, `additive` = 합집합 규칙, 버킷 규칙(아래 참조), 복호의 "이상음 우선" 구조.
- **우리 것**: ① SPRSOUND7 의 속성 분해표 ② 존재하지 않는 속성 조합이 예측됐을 때의 **복호 우선순위**
  ③ 연결 조합 목록을 7-class 로 넓힐 때의 나열 순서.

## 버킷 규칙은 발명이 아니라 유도다

저자 코드의 `classwise_cycle_list` 는 사이클을 클래스별 버킷에 넣는데, 규칙이
"0·1·2 는 해당 비트가 켜져 있으면, 3(both) 은 `original_label == 3` 이면" 이다.
이것은 **"사이클의 속성 벡터가 그 클래스의 속성 벡터를 포함(⊇)하면 넣는다"** 와 정확히 같다
(both `[0,1,1]` 만이 class 3 의 `[0,1,1]` 을 포함한다). 포함 규칙은 그대로 7-class 로 확장되고,
ICBHI 에서는 원 동작과 비트 단위로 같다 — `tests/unit/models/test_pcmcl_labelspace.py` 가 이를 고정한다.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


class LabelSpace:
    """클래스 인덱스와 속성(다중라벨) 벡터 사이의 변환 규약."""

    def __init__(
        self,
        name: str,
        class_names: Sequence[str],
        attr_names: Sequence[str],
        table: Sequence[Sequence[int]],
        decode_rules: Sequence[Tuple[Tuple[int, ...], int]],
        combination_builder=None,
    ):
        if len(table) != len(class_names):
            raise ValueError("table 행 수와 class_names 길이가 달라야 하지 않는다")
        if any(len(row) != len(attr_names) for row in table):
            raise ValueError("table 각 행의 길이는 attr_names 길이와 같아야 한다")
        self.name = name
        self.class_names = tuple(class_names)
        self.attr_names = tuple(attr_names)
        self.table = tuple(tuple(int(v) for v in row) for row in table)
        # 복호 규칙: (필요 속성 인덱스들, 결과 클래스) 를 순서대로 보고 **첫 일치가 이긴다**.
        # 어디에도 안 걸리면 0(정상). 저자 코드의 "이상음 우선" 구조를 그대로 옮긴 형태다.
        self.decode_rules = tuple((tuple(req), int(cls)) for req, cls in decode_rules)
        self._combination_builder = combination_builder
        self._table_t = torch.tensor(self.table, dtype=torch.float32)

    # ------------------------------------------------------------------ 기본
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_attrs(self) -> int:
        return len(self.attr_names)

    @property
    def profile_attrs(self) -> Tuple[int, int, int]:
        """PM 병리 프로필에 쓰는 속성 3개 — 두 공간 모두 (정상, 불연속, 연속) 이 0·1·2 다."""
        return (0, 1, 2)

    # ------------------------------------------------------------------ 부호화
    def to_multilabel(self, label: int) -> List[int]:
        """클래스 인덱스 → 속성 벡터(list[int]). 범위 밖 라벨은 조용히 넘기지 않는다."""
        idx = int(label)
        if not 0 <= idx < self.n_classes:
            raise ValueError(
                f"{self.name}: 클래스 인덱스 {idx} 는 0..{self.n_classes - 1} 밖이다. "
                "라벨 공간이 데이터셋과 맞는지 확인하라(잘못 맞추면 라벨이 조용히 왜곡된다)."
            )
        return list(self.table[idx])

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        """(B,) long → (B, n_attrs) float."""
        return self._table_t.to(labels.device)[labels.long()]

    # ------------------------------------------------------------------ 복호
    def decode_binary(self, bits: torch.Tensor) -> torch.Tensor:
        """(B, n_attrs) {0,1} → (B,) long. 규칙 순서대로 보고 첫 일치가 이긴다."""
        b = (bits > 0)
        out = torch.zeros(b.shape[0], dtype=torch.long, device=b.device)
        decided = torch.zeros(b.shape[0], dtype=torch.bool, device=b.device)
        for req, cls in self.decode_rules:
            hit = torch.ones_like(decided)
            for a in req:
                hit &= b[:, a]
            fresh = hit & ~decided
            out = torch.where(fresh, torch.full_like(out, cls), out)
            decided |= fresh
        return out

    def logits_to_class(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.decode_binary((torch.sigmoid(logits) > threshold).to(torch.int64))

    def logits_to_onehot(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """ICBHIScore(argmax 가정) 에 그대로 넣기 위한 (B, n_classes) one-hot."""
        y = self.logits_to_class(logits, threshold)
        return F.one_hot(y, num_classes=self.n_classes).to(logits.dtype)

    # ------------------------------------------------------------------ 버킷
    def buckets_for(self, multilabel: Sequence[int]) -> List[int]:
        """이 속성 벡터를 담을 클래스 버킷들 — 포함(⊇) 규칙. 모듈 docstring 참조."""
        return [k for k in range(self.n_classes)
                if all(multilabel[a] > 0 for a in range(self.n_attrs) if self.table[k][a] > 0)]

    # ------------------------------------------------------------------ 연결 조합
    def mixing_combinations(self, strategy: str = "additive") -> List[tuple]:
        """`(c1, c2[, c3], *결과_속성벡터, 확률)`. 원본과 같은 튜플 형식이라 소스 개수는 길이에서 되계산한다."""
        return self._combination_builder(strategy)

    def sources_in(self, combination: Sequence) -> int:
        """조합 튜플에서 소스 사이클 개수. 원본은 길이 6(2소스)·7(3소스) 로 분기했는데, 속성 수가
        3이 아니면 그 상수가 깨지므로 `길이 − 속성수 − 확률1` 로 되계산한다(ICBHI 에서 값이 같다)."""
        return len(combination) - self.n_attrs - 1

    def label_in(self, combination: Sequence) -> List[int]:
        n_src = self.sources_in(combination)
        return [int(v) for v in combination[n_src:n_src + self.n_attrs]]


# ====================================================================== ICBHI 4-class (저자 정의)
def _icbhi_combinations(strategy: str = "additive") -> List[tuple]:
    """원본 `_get_multilabel_mixing_combinations` 와 같은 순서·내용(random.choice 인덱스가 같아야 한다).

    `pcmcl_concat.multilabel_mixing_combinations` 가 원본 이식이라 그대로 부른다 — 목록을 두 벌 두지 않는다."""
    from rsc_toolkit.datasets.pcmcl_concat import multilabel_mixing_combinations

    return multilabel_mixing_combinations(strategy)


ICBHI4 = LabelSpace(
    name="icbhi4",
    class_names=("normal", "crackle", "wheeze", "both"),
    attr_names=("normal", "crackle", "wheeze"),
    table=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],
    # conversion.py convert_original(additive) 와 같은 결과: c&w→3, c→1, w→2, 그 외→0
    decode_rules=[((1, 2), 3), ((1,), 1), ((2,), 2)],
    combination_builder=_icbhi_combinations,
)


# ====================================================================== SPRSound 7-class (툴킷 정의)
# 인덱스: 0 Normal · 1 Fine Crackle · 2 Wheeze · 3 Coarse Crackle · 4 Wheeze+Crackle · 5 Rhonchi · 6 Stridor
# 속성   : 0 normal · 1 discont(불연속=crackle 축) · 2 cont(연속=wheeze 축) · 3 coarse · 4 rhonchi · 5 stridor
_SPRS_TABLE = [
    [1, 0, 0, 0, 0, 0],   # Normal
    [0, 1, 0, 0, 0, 0],   # Fine Crackle      = 불연속
    [0, 0, 1, 0, 0, 0],   # Wheeze            = 연속
    [0, 1, 0, 1, 0, 0],   # Coarse Crackle    = 불연속 + 거친
    [0, 1, 1, 0, 0, 0],   # Wheeze+Crackle    = 불연속 + 연속   (ICBHI 의 both 와 같은 자리)
    [0, 0, 1, 0, 1, 0],   # Rhonchi           = 연속 + rhonchi
    [0, 0, 1, 0, 0, 1],   # Stridor           = 연속 + stridor
]


def _sprsound_combinations(strategy: str = "additive") -> List[tuple]:
    """7-class 의 연결 조합 — **모든 비순서 쌍(자기 자신 포함) 28개**, 라벨은 속성 합집합.

    저자의 ICBHI 목록도 정확히 이 규칙이다: 4-class 의 비순서 쌍 C(4+1,2)=10 개 전부이고,
    `additive` 라벨이 두 클래스 속성 벡터의 합집합이다(예: (0,3) → normal ∪ both = [1,1,1]).
    확인은 `tests/unit/models/test_pcmcl_labelspace.py::test_icbhi_pairs_are_union_of_attributes`.

    ICBHI 목록에는 3-소스 조합이 하나 있다 — `(normal, crackle, wheeze)`, 즉 "기본 속성 하나씩".
    속성이 6개인 SPRSound 에는 그에 대응하는 **유일한** 3-소스 조합이 없어 넣지 않았다.
    (연결 길이가 2 로 고정된다는 뜻이고, 이것이 ICBHI 레시피와 다른 유일한 지점이다.)
    """
    if strategy not in {"additive", "dominant"}:
        raise ValueError(f"mixing_label_strategy must be additive|dominant, got {strategy!r}")
    if strategy == "dominant":
        raise NotImplementedError(
            "sprsound7 은 additive(합집합) 만 정의했다 — dominant 는 '어느 쪽이 우세한가' 를 "
            "7-class 에서 새로 정해야 하고, 그 규칙은 논문에도 저자 코드에도 없다."
        )
    combos: List[tuple] = []
    for i in range(len(_SPRS_TABLE)):
        for j in range(i, len(_SPRS_TABLE)):
            union = [max(a, b) for a, b in zip(_SPRS_TABLE[i], _SPRS_TABLE[j])]
            combos.append((i, j, *union, 1.0))
    return combos


SPRSOUND7 = LabelSpace(
    name="sprsound7",
    class_names=("normal", "fine_crackle", "wheeze", "coarse_crackle", "wheeze_crackle", "rhonchi", "stridor"),
    attr_names=("normal", "discont", "cont", "coarse", "rhonchi", "stridor"),
    table=_SPRS_TABLE,
    # 복호 우선순위(**우리가 정한 것**) — ICBHI 와 같은 구조: 복합이 단일을 이기고, 이상음이 정상을 이긴다.
    #   ① 불연속+연속 → Wheeze+Crackle   ② 불연속+거친 → Coarse   ③ 불연속 → Fine
    #   ④ 연속+rhonchi → Rhonchi         ⑤ 연속+stridor → Stridor  ⑥ 연속 → Wheeze   ⑦ 그 외 → Normal
    # rhonchi 와 stridor 가 동시에 켜지면 rhonchi 가 이긴다(자의적 — 각주로 밝힌다).
    decode_rules=[((1, 2), 4), ((1, 3), 3), ((1,), 1), ((2, 4), 5), ((2, 5), 6), ((2,), 2)],
    combination_builder=_sprsound_combinations,
)


_SPACES: Dict[str, LabelSpace] = {"icbhi4": ICBHI4, "sprsound7": SPRSOUND7}


def get_label_space(name: str) -> LabelSpace:
    try:
        return _SPACES[name]
    except KeyError:
        raise ValueError(f"label_space must be one of {sorted(_SPACES)}, got {name!r}") from None
