# Portions of this file are ported from SG-SCL (https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
meta (domain) labels -- auxiliary labels used by sgscl-family (DAT, SG-SCL), PAFA and SCL-metadata training.

Values of `data.params.meta_label_mode` are names in `rsc_toolkit.registries.META_LABELS`. A registered object is a
**factory** (called without arguments, it returns an `item -> int` function). The 4 built-in modes are defined below and declared
lazily in the hub; third parties attach theirs with the decorator:

    from rsc_toolkit.registries import META_LABELS
    @META_LABELS.register("hospital")
    def build_hospital():
        return lambda item: int(item["meta"]["hospital"])

Follows the mapping of get_meta_infor() in the original sgscl util/icbhi_util.py. The modes actually used by the papers/scripts are only
'dev' (4 stethoscope devices), 'patient' (PAFA) and 'sa'/'sa_sprs' (SCL-metadata).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

# Original icbhi_dataset.py:41 -- device_to_id (same order as meta_cls_list)
ICBHI_DEVICE_TO_ID = {"Meditron": 0, "LittC2SE": 1, "Litt3200": 2, "AKGC417L": 3}

MetaLabelFn = Callable[[Dict[str, Any]], int]


class UnknownMetaLabelMode(KeyError, NotImplementedError):
    """Unknown meta_label_mode. (Both the registry's KeyError and the NotImplementedError expected by legacy callers.)"""

    def __str__(self) -> str:  # KeyError wraps the message in repr, so show the message as-is
        return str(self.args[0]) if self.args else ""


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _is_adult(meta: Mapping[str, Any]) -> bool:
    """ICBHI meta: adults have adult_BMI, children have child_weight/child_height; if neither is present, age 19 is the threshold."""
    if _num(meta.get("adult_BMI")) is not None:
        return True
    if _num(meta.get("child_weight")) is not None or _num(meta.get("child_height")) is not None:
        return False
    age = _num(meta.get("age"))
    # A patient with no metadata at all (ICBHI 223) is M/A (=0) in the original metadata.csv -> missing is treated as adult
    return age is None or age >= 19


# ── built-in mode factories (declared lazily in registries.META_LABELS) ──────────────────────────────────
def build_dev() -> MetaLabelFn:
    """mode="dev": ICBHI stethoscope device -> {Meditron:0, LittC2SE:1, Litt3200:2, AKGC417L:3}."""

    def fn(item: Dict[str, Any]) -> int:
        device = str(item["meta"]["device"])
        try:
            return ICBHI_DEVICE_TO_ID[device]
        except KeyError:
            raise KeyError(
                f"unknown device {device!r} (sample_id={item.get('sample_id')!r}); "
                f"expected: {sorted(ICBHI_DEVICE_TO_ID)}"
            ) from None

    return fn


def build_patient() -> MetaLabelFn:
    """mode="patient": PAFA (wa976/PAFA icbhi_dataset.py) -- the patient number at the start of the file name, used as an integer."""

    def fn(item: Dict[str, Any]) -> int:
        return int(item["meta"]["patient_id"])

    return fn


def build_sa() -> MetaLabelFn:
    """mode="sa": sa_class_num from SCL-metadata (ilyassmoummad/scl_icbhi2017) metadata.csv.

    sex (M/F) x age group (A=adult, C=child) -> M/A 0, F/A 1, F/C 2, M/C 3.
    A/C follows the adult (BMI recorded) / child (height and weight recorded) distinction in ICBHI meta (consistent with the age-19 threshold).
    Missing (patient 223: no sex or age) is M/A -> 0, as in the original CSV. Verified against all 6898 cycles.
    """

    def fn(item: Dict[str, Any]) -> int:
        m = item["meta"]
        sex = m.get("sex")
        female = sex in ("F", "f", 1, 1.0, "1", "1.0")
        adult = _is_adult(m)
        if adult:
            return 1 if female else 0
        return 2 if female else 3

    return fn


def build_sa_sprs() -> MetaLabelFn:
    """mode="sa_sprs": sa_class_num from the SCL-metadata SPRS metadata.csv.

    gender(0/1) x age_group(0: <5 years, 1: >=5 years) -> gender*2 + age_group
    (train distribution 2004/1558/1527/1567; age_group boundary 4.9 | 5.0 -- verified against the full original CSV)
    """

    def fn(item: Dict[str, Any]) -> int:
        m = item["meta"]
        gender = int(float(m.get("gender", 0)))
        age = _num(m.get("age"))
        age_group = 1 if (age is not None and age >= 5.0) else 0
        return gender * 2 + age_group

    return fn


def build_meta_label_fn(mode: str) -> MetaLabelFn:
    """
    Build a function that extracts the integer meta label from an IndexDataset item (with meta dict).
    mode is a name in META_LABELS (built-in: dev, patient, sa, sa_sprs, or a plugin-registered name).
    """
    from rsc_toolkit.registries import META_LABELS

    try:
        factory = META_LABELS.get(mode)
    except KeyError:
        raise UnknownMetaLabelMode(
            f"meta_label_mode={mode!r} is not supported. Available: {sorted(META_LABELS)}. "
            "Register a factory with META_LABELS.register if needed (see the original get_meta_infor mapping)."
        ) from None
    return factory()
