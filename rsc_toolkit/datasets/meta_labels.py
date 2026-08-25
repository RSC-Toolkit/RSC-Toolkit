"""
meta(도메인) 라벨 — sgscl 계열(DAT·SG-SCL) 학습이 쓰는 보조 라벨.

원본 sgscl util/icbhi_util.py get_meta_infor() 의 매핑을 따른다. 논문·스크립트가
실제로 쓰는 것은 'dev'(청진기 장비 4종)뿐이라 우선 그것만 지원한다.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

# 원본 icbhi_dataset.py:41 — device_to_id (meta_cls_list 순서와 동일)
ICBHI_DEVICE_TO_ID = {"Meditron": 0, "LittC2SE": 1, "Litt3200": 2, "AKGC417L": 3}

MetaLabelFn = Callable[[Dict[str, Any]], int]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _is_adult(meta: Mapping[str, Any]) -> bool:
    """ICBHI 메타: 성인은 adult_BMI, 소아는 child_weight/child_height 가 기재된다; 둘 다 없으면 나이 19세 기준."""
    if _num(meta.get("adult_BMI")) is not None:
        return True
    if _num(meta.get("child_weight")) is not None or _num(meta.get("child_height")) is not None:
        return False
    age = _num(meta.get("age"))
    # 메타가 전혀 없는 환자(ICBHI 223)는 원본 metadata.csv 가 M·A(=0) 로 둔다 → 결측은 성인으로
    return age is None or age >= 19


def build_meta_label_fn(mode: str) -> MetaLabelFn:
    """
    IndexDataset 아이템(meta dict 포함)에서 meta 라벨 정수를 뽑는 함수를 만든다.

    mode="dev": ICBHI 청진기 장비 → {Meditron:0, LittC2SE:1, Litt3200:2, AKGC417L:3}
    (원본의 age/sex/loc/label 모드는 대상 논문이 안 써서 아직 안 만든다.)
    """
    if mode == "dev":

        def fn(item: Dict[str, Any]) -> int:
            device = str(item["meta"]["device"])
            try:
                return ICBHI_DEVICE_TO_ID[device]
            except KeyError:
                raise KeyError(
                    f"알 수 없는 device {device!r} (sample_id={item.get('sample_id')!r}); "
                    f"기대값: {sorted(ICBHI_DEVICE_TO_ID)}"
                ) from None

        return fn

    if mode == "patient":
        # PAFA(wa976/PAFA icbhi_dataset.py): 파일명 앞 환자 번호를 정수 그대로 쓴다
        def fn(item: Dict[str, Any]) -> int:
            return int(item["meta"]["patient_id"])

        return fn

    if mode == "sa":
        # SCL-metadata(ilyassmoummad/scl_icbhi2017) metadata.csv 의 sa_class_num:
        # 성별(M/F) x 연령군(A=성인, C=소아) → M·A 0, F·A 1, F·C 2, M·C 3.
        # A/C 는 ICBHI 메타의 성인(BMI 기재)/소아(신장·체중 기재) 구분 (나이 19세 경계와 일치).
        # 결측(환자 223: 성별·나이 없음)은 원본 CSV 대로 M·A → 0. 전 6898 cycle 대조로 확인.
        def fn(item: Dict[str, Any]) -> int:
            m = item["meta"]
            sex = m.get("sex")
            female = sex in ("F", "f", 1, 1.0, "1", "1.0")
            adult = _is_adult(m)
            if adult:
                return 1 if female else 0
            return 2 if female else 3

        return fn

    if mode == "sa_sprs":
        # SCL-metadata SPRS metadata.csv 의 sa_class_num: gender(0/1) x age_group(0: <5세, 1: ≥5세) → gender*2 + age_group
        # (train 분포 2004/1558/1527/1567; age_group 경계 4.9 | 5.0 — 원본 CSV 전수 대조로 확인)
        def fn(item: Dict[str, Any]) -> int:
            m = item["meta"]
            gender = int(float(m.get("gender", 0)))
            age = _num(m.get("age"))
            age_group = 1 if (age is not None and age >= 5.0) else 0
            return gender * 2 + age_group

        return fn

    raise NotImplementedError(
        f"meta_label_mode={mode!r} 는 아직 지원하지 않는다 (지원: 'dev', 'patient', 'sa', 'sa_sprs'). "
        "필요해지는 논문이 정해지면 원본 get_meta_infor 매핑대로 추가할 것."
    )
