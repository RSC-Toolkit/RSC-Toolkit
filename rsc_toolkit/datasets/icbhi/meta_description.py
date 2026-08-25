"""
BTS 논문(Interspeech 2024)의 메타데이터 → 텍스트 설명 — 공식 코드 충실 이식 (P0).

출처: https://github.com/kaen2891/bts util/meta_description.py generate_meta_description()
      + util/icbhi_util.py get_meta_text_descriptions() (학습·기본 평가 경로).

원본 그대로 유지하는 quirk:
- 나이는 19세 이상이면 'adult', 아니면 'pediatric'. 나이 결측(원본은 NaN→-1)도 -1 < 19
  라서 'pediatric' 이 된다 — 원본 학습/평가 경로에 unknown 처리가 없다.
- 성별은 0(M)→'male', 그 외→'female'.
- 원본의 테스트 전용 ablation 플래그(test_drop_key / test_unknown_all / test_bmi /
  test_wrong_label)는 논문 재현 수치와 무관해 이식하지 않는다.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

# 원본 icbhi_dataset.py 의 7필드 meta 텐서: age, sex, adult_BMI, child_weight, child_height, chest_location, device
LOC_ID_TO_STR = {
    0: "trachea", 1: "left anterior chest", 2: "right anterior chest", 3: "left posterior chest",
    4: "right posterior chest", 5: "left lateral chest", 6: "right lateral chest",
}
LOC_CODE_TO_ID = {"Tc": 0, "Al": 1, "Ar": 2, "Pl": 3, "Pr": 4, "Ll": 5, "Lr": 6}
DEV_ID_TO_STR = {0: "Meditron", 1: "LittC2SE", 2: "Litt3200", 3: "AKGC417L"}
DEV_TO_ID = {v: k for k, v in DEV_ID_TO_STR.items()}
ADULT_AGE = 19

META_MODES = (
    "age", "sex", "loc", "dev", "age_sex", "age_loc", "age_dev", "sex_loc", "sex_dev", "loc_dev",
    "age_sex_loc", "age_sex_dev", "age_loc_dev", "sex_loc_dev", "all",
)


def generate_meta_description(age=None, sex=None, loc=None, dev=None):
    """원본 meta_description.py:6-69 그대로."""
    # set up the prefix
    if loc or dev:
        output_sent = "This sound was recorded"
    else:
        output_sent = "This patient is"

    # prepare the main descriptions
    if loc:
        s_loc = "an unknown region" if loc == "unknown" else f"the {loc}"
        s_loc = f"from {s_loc}"

    if age and sex:
        if age == "unknown":
            s_age_sex = f"a {sex} patient of unknown age"
        elif sex == "unknown":
            s_age_sex = "an " if age == "adult" else "a "
            s_age_sex += f"{age} patient of unknown sex"
        elif age == "pediatric":
            s_age_sex = f"a {sex} {age} patient"
        else:
            s_age_sex = f"an {age} {sex} patient"
    elif age:
        if age == "unknown":
            s_age_sex = "a patient of unknown age"
        else:
            s_age_sex = "an " if age == "adult" else "a "
            s_age_sex += f"{age} patient"
    elif sex:
        if sex == "unknown":
            s_age_sex = "a patient of unknown sex"
        else:
            s_age_sex = f"a {sex} patient"

    if dev:
        if dev == "unknown":
            s_dev = "an unknown device"
        else:
            s_dev_prefix = "an" if dev == "AKGC417L" else "a"
            s_dev_suffix = "microphone" if dev == "AKGC417L" else "stethoscope"
            s_dev = f"{s_dev_prefix} {dev} {s_dev_suffix}"

    # combine the descriptions
    if "s_loc" in locals():
        output_sent += f" {s_loc}"
        if "s_age_sex" in locals():
            s_age_sex = s_age_sex.replace("of", "with")

    if "s_age_sex" in locals():
        if "s_loc" not in locals() and "s_dev" not in locals():
            output_sent += f" {s_age_sex}"
        elif "s_loc" in locals():
            output_sent += f" of {s_age_sex}"
        else:
            output_sent += f" from {s_age_sex}"

    if "s_age_sex" in locals() and "s_dev" in locals():
        output_sent += f", using {s_dev}"
    elif "s_dev" in locals():
        output_sent += f" with {s_dev}"

    output_sent += "."

    return output_sent


def meta_fields_from_meta7(meta7: Sequence[float]) -> Dict[str, str]:
    """원본 get_meta_text_descriptions 의 정수→문자열 변환 (icbhi_util.py:130-171)."""
    age, sex, loc, dev = int(meta7[0]), int(meta7[1]), int(meta7[5]), int(meta7[6])
    return {
        "age": "adult" if age >= ADULT_AGE else "pediatric",
        "sex": "male" if sex == 0 else "female",
        "loc": LOC_ID_TO_STR[loc],
        "dev": DEV_ID_TO_STR[dev],
    }


def select_meta_fields(fields: Mapping[str, str], meta_mode: str) -> Dict[str, Optional[str]]:
    """원본의 meta_mode 별 meta_dict 구성 (icbhi_util.py:209-239): 포함 안 되는 항목은 None."""
    if meta_mode not in META_MODES:
        raise ValueError(f"meta_mode must be one of {META_MODES}, got {meta_mode!r}")
    keep = {"age", "sex", "loc", "dev"} if meta_mode == "all" else set(meta_mode.split("_"))
    return {k: (fields[k] if k in keep else None) for k in ("age", "sex", "loc", "dev")}


def describe_meta7(meta7: Sequence[float], meta_mode: str = "all") -> str:
    """저자 전처리 .pt 의 7필드 meta 텐서 → 텍스트 설명 (학습·기본 평가 경로와 동일)."""
    return generate_meta_description(**select_meta_fields(meta_fields_from_meta7(meta7), meta_mode))


def meta7_from_index_meta(meta: Mapping[str, Any]) -> list:
    """
    툴킷 ICBHI 인덱스의 meta dict(age, sex, adult_BMI, child_weight, child_height,
    chest_location, device) → 원본과 같은 7필드 벡터. 결측은 원본처럼 -1.
    """
    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return -1.0
        return -1.0 if f != f else f  # NaN → -1

    sex = str(meta.get("sex", ""))
    return [
        num(meta.get("age")),
        0.0 if sex == "M" else 1.0,
        num(meta.get("adult_BMI")),
        num(meta.get("child_weight")),
        num(meta.get("child_height")),
        float(LOC_CODE_TO_ID[str(meta["chest_location"])]),
        float(DEV_TO_ID[str(meta["device"])]),
    ]
