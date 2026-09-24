# Portions of this file are ported from PC-MCL (https://github.com/wa976/PC-MCL), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
**Label space** of PC-MCL: class index ↔ attribute vector conversion, concatenation combinations and bucket rules in one place.

## Why it is needed

MCL in PC-MCL (arXiv 2601.17080) depends not on "how many classes" but on **"whether labels can be decomposed into attributes"**.
ICBHI 4-class happens to be two binary attributes (crackle yes/no × wheeze yes/no), so the authors' code trains not a 4-way classifier but
a 3-label `(normal, crackle, wheeze)` one: `both` becomes `[0,1,1]`, so the crackle detector
learns from both class 1 and class 3. This **attribute sharing** is the essence of MCL.

So moving to another dataset requires not a different head size but **an attribute decomposition of that dataset**.
This module holds that decomposition.

## Two label spaces

| | ICBHI4 (defined by the authors) | SPRSOUND7 (defined by the toolkit; extension) |
|---|---|---|
| classes | 4 | 7 |
| attributes | 3 `normal · crackle · wheeze` | 6 `normal · discont · cont · coarse · rhonchi · stridor` |
| source | `util/dataset_pcmcl.py _convert_to_multilabel` | **defined by the toolkit**; not in the paper or the authors' code |

`discont` (discontinuous adventitious sound) continues ICBHI's `crackle` axis, and `cont` (continuous adventitious sound) continues the `wheeze` axis.
So attributes 0·1·2 play the same role in both spaces, and the pathology-profile classification of PM (patient matching) works without code changes.

## What comes from the authors and what comes from the toolkit

- **Authors**: the ICBHI decomposition, `additive` = union rule, the bucket rule (see below), the "adventitious sound first" structure of decoding.
- **Toolkit**: (1) the SPRSOUND7 attribute decomposition table, (2) the **decoding priority** when a non-existent attribute combination is predicted,
  (3) the enumeration order when extending the concatenation combination list to 7 classes.

## The bucket rule is derived, not invented

`classwise_cycle_list` in the authors' code puts cycles into per-class buckets with the rule
"0·1·2 if the corresponding bit is on, 3 (both) if `original_label == 3`".
This is exactly **"put the cycle in if its attribute vector contains (⊇) the class's attribute vector"**
(only both `[0,1,1]` contains class 3's `[0,1,1]`). The containment rule extends directly to 7 classes and
is bit-identical to the original behavior on ICBHI; `tests/unit/models/test_pcmcl_labelspace.py` pins this down.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


class LabelSpace:
    """Conversion convention between class indices and attribute (multi-label) vectors."""

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
            raise ValueError("the number of table rows must equal the length of class_names")
        if any(len(row) != len(attr_names) for row in table):
            raise ValueError("each table row must have the same length as attr_names")
        self.name = name
        self.class_names = tuple(class_names)
        self.attr_names = tuple(attr_names)
        self.table = tuple(tuple(int(v) for v in row) for row in table)
        # Decoding rules: (required attribute indices, resulting class) checked in order; **the first match wins**.
        # If nothing matches, 0 (normal). Mirrors the "adventitious sound first" structure of the authors' code.
        self.decode_rules = tuple((tuple(req), int(cls)) for req, cls in decode_rules)
        self._combination_builder = combination_builder
        self._table_t = torch.tensor(self.table, dtype=torch.float32)
        # encode is called at every validation step, so the table is moved once per device and cached
        # (LabelSpace is not an nn.Module, so it cannot register a buffer).
        self._table_by_device: dict = {self._table_t.device: self._table_t}

    # ------------------------------------------------------------------ basics
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_attrs(self) -> int:
        return len(self.attr_names)

    @property
    def profile_attrs(self) -> Tuple[int, int, int]:
        """The three attributes used for the PM pathology profile; in both spaces (normal, discontinuous, continuous) are 0·1·2."""
        return (0, 1, 2)

    # ------------------------------------------------------------------ encoding
    def to_multilabel(self, label: int) -> List[int]:
        """Class index → attribute vector (list[int]). Out-of-range labels are never passed through silently."""
        idx = int(label)
        if not 0 <= idx < self.n_classes:
            raise ValueError(
                f"{self.name}: class index {idx} is outside 0..{self.n_classes - 1}. "
                "Check that the label space matches the dataset (a mismatch silently corrupts labels)."
            )
        return list(self.table[idx])

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        """(B,) long → (B, n_attrs) float."""
        table = self._table_by_device.get(labels.device)
        if table is None:
            table = self._table_t.to(labels.device)
            self._table_by_device[labels.device] = table
        return table[labels.long()]

    # ------------------------------------------------------------------ decoding
    def decode_binary(self, bits: torch.Tensor) -> torch.Tensor:
        """(B, n_attrs) {0,1} → (B,) long. Rules are checked in order; the first match wins."""
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
        """(B, n_classes) one-hot to feed ICBHIScore (which assumes argmax) directly."""
        y = self.logits_to_class(logits, threshold)
        return F.one_hot(y, num_classes=self.n_classes).to(logits.dtype)

    # ------------------------------------------------------------------ buckets
    def buckets_for(self, multilabel: Sequence[int]) -> List[int]:
        """Class buckets that receive this attribute vector (containment ⊇ rule). See the module docstring."""
        return [k for k in range(self.n_classes)
                if all(multilabel[a] > 0 for a in range(self.n_attrs) if self.table[k][a] > 0)]

    # ------------------------------------------------------------------ concatenation combinations
    def mixing_combinations(self, strategy: str = "additive") -> List[tuple]:
        """`(c1, c2[, c3], *resulting_attribute_vector, probability)`. Same tuple format as the original code, so the source count is derived from the length."""
        return self._combination_builder(strategy)

    def sources_in(self, combination: Sequence) -> int:
        """Number of source cycles in a combination tuple. The original code branches on length 6 (2 sources) / 7 (3 sources), but those
        constants break when the attribute count is not 3, so it is derived as `length − n_attrs − 1 (probability)` (same values on ICBHI)."""
        return len(combination) - self.n_attrs - 1

    def label_in(self, combination: Sequence) -> List[int]:
        n_src = self.sources_in(combination)
        return [int(v) for v in combination[n_src:n_src + self.n_attrs]]


# ====================================================================== ICBHI 4-class (defined by the authors)
def _icbhi_combinations(strategy: str = "additive") -> List[tuple]:
    """Same order and content as the original `_get_multilabel_mixing_combinations` (random.choice indices must match).

    `pcmcl_concat.multilabel_mixing_combinations` is the port of the original, so it is called directly; the list is not duplicated."""
    from rsc_toolkit.datasets.pcmcl_concat import multilabel_mixing_combinations

    return multilabel_mixing_combinations(strategy)


ICBHI4 = LabelSpace(
    name="icbhi4",
    class_names=("normal", "crackle", "wheeze", "both"),
    attr_names=("normal", "crackle", "wheeze"),
    table=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],
    # same result as conversion.py convert_original(additive): c&w→3, c→1, w→2, otherwise→0
    decode_rules=[((1, 2), 3), ((1,), 1), ((2,), 2)],
    combination_builder=_icbhi_combinations,
)


# ====================================================================== SPRSound 7-class (defined by the toolkit)
# indices   : 0 Normal · 1 Fine Crackle · 2 Wheeze · 3 Coarse Crackle · 4 Wheeze+Crackle · 5 Rhonchi · 6 Stridor
# attributes: 0 normal · 1 discont(discontinuous = crackle axis) · 2 cont(continuous = wheeze axis) · 3 coarse · 4 rhonchi · 5 stridor
_SPRS_TABLE = [
    [1, 0, 0, 0, 0, 0],   # Normal
    [0, 1, 0, 0, 0, 0],   # Fine Crackle      = discont
    [0, 0, 1, 0, 0, 0],   # Wheeze            = cont
    [0, 1, 0, 1, 0, 0],   # Coarse Crackle    = discont + coarse
    [0, 1, 1, 0, 0, 0],   # Wheeze+Crackle    = discont + cont   (same position as ICBHI's both)
    [0, 0, 1, 0, 1, 0],   # Rhonchi           = cont + rhonchi
    [0, 0, 1, 0, 0, 1],   # Stridor           = cont + stridor
]


def _sprsound_combinations(strategy: str = "additive") -> List[tuple]:
    """Concatenation combinations for 7 classes: **all 28 unordered pairs (including self-pairs)**, label = attribute union.

    The authors' ICBHI list follows exactly this rule: all C(4+1,2)=10 unordered pairs of the 4 classes,
    with the `additive` label being the union of the two classes' attribute vectors (e.g. (0,3) → normal ∪ both = [1,1,1]).
    Checked by `tests/unit/models/test_pcmcl_labelspace.py::test_icbhi_pairs_are_union_of_attributes`.

    The ICBHI list has one 3-source combination, `(normal, crackle, wheeze)`, i.e. "one of each basic attribute".
    SPRSound with 6 attributes has no **unique** corresponding 3-source combination, so none is included.
    (The concatenation length is therefore fixed at 2; this is the only difference from the ICBHI recipe.)
    """
    if strategy not in {"additive", "dominant"}:
        raise ValueError(f"mixing_label_strategy must be additive|dominant, got {strategy!r}")
    if strategy == "dominant":
        raise NotImplementedError(
            "sprsound7 defines only additive (union); dominant would need a new definition of 'which side dominates' "
            "for 7 classes, and that rule is in neither the paper nor the authors' code."
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
    # Decoding priority (**defined by the toolkit**), same structure as ICBHI: combined beats single, adventitious beats normal.
    #   ① discont+cont → Wheeze+Crackle   ② discont+coarse → Coarse   ③ discont → Fine
    #   ④ cont+rhonchi → Rhonchi          ⑤ cont+stridor → Stridor    ⑥ cont → Wheeze   ⑦ otherwise → Normal
    # If rhonchi and stridor are both on, rhonchi wins (an arbitrary choice).
    decode_rules=[((1, 2), 4), ((1, 3), 3), ((1,), 1), ((2, 4), 5), ((2, 5), 6), ((2,), 2)],
    combination_builder=_sprsound_combinations,
)


_SPACES: Dict[str, LabelSpace] = {"icbhi4": ICBHI4, "sprsound7": SPRSOUND7}


def get_label_space(name: str) -> LabelSpace:
    try:
        return _SPACES[name]
    except KeyError:
        raise ValueError(f"label_space must be one of {sorted(_SPACES)}, got {name!r}") from None
