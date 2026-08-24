"""
meta(도메인) 라벨 — sgscl 계열(DAT·SG-SCL) 학습이 쓰는 보조 라벨.

원본 sgscl util/icbhi_util.py get_meta_infor() 의 매핑을 따른다. 논문·스크립트가
실제로 쓰는 것은 'dev'(청진기 장비 4종)뿐이라 우선 그것만 지원한다.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

# 원본 icbhi_dataset.py:41 — device_to_id (meta_cls_list 순서와 동일)
ICBHI_DEVICE_TO_ID = {"Meditron": 0, "LittC2SE": 1, "Litt3200": 2, "AKGC417L": 3}

MetaLabelFn = Callable[[Dict[str, Any]], int]


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

    raise NotImplementedError(
        f"meta_label_mode={mode!r} 는 아직 지원하지 않는다 (지원: 'dev'). "
        "필요해지는 논문이 정해지면 원본 get_meta_infor 매핑대로 추가할 것."
    )
