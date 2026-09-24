# Portions of this file are ported from BTS (https://github.com/kaen2891/bts), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Metadata -> text description of the BTS paper (Interspeech 2024) -- exact port of the official code (verified against goldens).

Source: https://github.com/kaen2891/bts util/meta_description.py generate_meta_description()
      + util/icbhi_util.py get_meta_text_descriptions() (training and default evaluation path).

Quirks kept as in the original code:
- Age 19 or older is 'adult', otherwise 'pediatric'. Missing age (NaN -> -1 in the original) also satisfies -1 < 19
  and becomes 'pediatric' -- the original training/evaluation path has no unknown handling.
- Sex 0 (M) -> 'male', anything else -> 'female'.
- The original test-only ablation flags (test_drop_key / test_unknown_all / test_bmi /
  test_wrong_label) do not affect the paper's reproduction numbers and are not ported.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

# 7-field meta tensor of the original icbhi_dataset.py: age, sex, adult_BMI, child_weight, child_height, chest_location, device
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
    """Same as the original meta_description.py:6-69."""
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
    """Integer -> string conversion of the original get_meta_text_descriptions (icbhi_util.py:130-171)."""
    age, sex, loc, dev = int(meta7[0]), int(meta7[1]), int(meta7[5]), int(meta7[6])
    return {
        "age": "adult" if age >= ADULT_AGE else "pediatric",
        "sex": "male" if sex == 0 else "female",
        "loc": LOC_ID_TO_STR[loc],
        "dev": DEV_ID_TO_STR[dev],
    }


def select_meta_fields(fields: Mapping[str, str], meta_mode: str) -> Dict[str, Optional[str]]:
    """meta_dict construction per meta_mode as in the original (icbhi_util.py:209-239): excluded fields are None."""
    if meta_mode not in META_MODES:
        raise ValueError(f"meta_mode must be one of {META_MODES}, got {meta_mode!r}")
    keep = {"age", "sex", "loc", "dev"} if meta_mode == "all" else set(meta_mode.split("_"))
    return {k: (fields[k] if k in keep else None) for k in ("age", "sex", "loc", "dev")}


def describe_meta7(meta7: Sequence[float], meta_mode: str = "all") -> str:
    """7-field meta tensor from the authors' preprocessed .pt -> text description (same as the training and default evaluation path)."""
    return generate_meta_description(**select_meta_fields(meta_fields_from_meta7(meta7), meta_mode))


def meta7_from_index_meta(meta: Mapping[str, Any]) -> list:
    """
    Toolkit ICBHI index meta dict (age, sex, adult_BMI, child_weight, child_height,
    chest_location, device) -> the same 7-field vector as the original. Missing values are -1, as in the original.
    """
    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return -1.0
        return -1.0 if f != f else f  # NaN -> -1

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
