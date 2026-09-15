import inspect
from typing import Any, Dict, List

import lightning as L

from rsc_toolkit.registries import DATAMODULES

# configs/data_defaults/base.yaml 이 `data_defaults@data: base` 로 주입하는 키 — index 계열 datamodule 의 인자다.
# 엄격한 시그니처의 datamodule(bts_precomputed·respirenet 등)과 `data=<name>` override 로 조합되면 이 키가 남는다.
_DATA_DEFAULTS_KEYS = ("desired_length_sec", "pad_type", "specaugment")


def unknown_datamodule_kwargs(cls: type, params: Dict[str, Any]) -> List[str]:
    """`cls(**params)` 가 받지 못할 키. **kwargs 를 받는 클래스면 빈 목록(자기 책임)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):      # C 확장·시그니처 없는 호출 가능 객체
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return []
    accepted = {n for n, p in sig.parameters.items()
                if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return sorted(k for k in params if k not in accepted)


def build_datamodule(cfg: Dict[str, Any]) -> L.LightningDataModule:
    """
    cfg example (configs/data/*.yaml):
      {"name": "icbhi",                       # 데이터셋 이름 (로그·run 이름용; 조회 키가 아니다)
       "params": {"datamodule_name": "index",  # 조회 키 — 또는 최상위 "datamodule": "index"
                  "batch_size": 32, "num_workers": 4, ...}}

    datamodule 은 `params.datamodule_name` (기존 규약) 또는 최상위 `datamodule` 로 지정한다.
    params 는 복사해서 넘기므로 cfg 는 변하지 않는다 (예전엔 pop 으로 파괴해 cfg 재사용이 KeyError 였다).
    누락·미등록 오류는 실제 실패한 키(datamodule_name)를 지목한다.
    """
    params = dict(cfg.get("params") or {})
    name = params.pop("datamodule_name", None) or cfg.get("datamodule")
    if name is None:
        raise KeyError(
            "data config 에 datamodule 지정이 없다: data.params.datamodule_name "
            f"(또는 data.datamodule) 를 넣어라. Available: {sorted(DATAMODULES)}"
        )
    try:
        cls = DATAMODULES.get(name)
    except KeyError:
        raise KeyError(
            f"Unknown datamodule '{name}' (data.params.datamodule_name). Available: {sorted(DATAMODULES)}"
        ) from None
    unknown = unknown_datamodule_kwargs(cls, params)
    if unknown:
        # 예전엔 cls(**params) 의 TypeError("unexpected keyword argument ...") 가 그대로 올라와 원인(어느 config 가
        # 그 키를 넣었는지)을 알 수 없었다 (REVIEW 2026-09-10 §7 P1 — P2-10 config 스키마의 첫 사례).
        injected = [k for k in unknown if k in _DATA_DEFAULTS_KEYS]
        hint = ""
        if injected:
            hint = (f" {injected} 는 `data_defaults@data: base` 가 주입하는 index 계열 인자다 — 이 datamodule 과 조합하려면 "
                    f"최상위 config 의 defaults 에서 data_defaults 를 빼거나 CLI 로 제거하라: "
                    + " ".join(f"~data.params.{k}" for k in injected) + ".")
        raise TypeError(
            f"datamodule '{name}' ({cls.__module__}.{cls.__qualname__}) 은 data.params 의 키 {unknown} 를 받지 않는다."
            f"{hint} (오타라면 config 의 키 이름을 확인하라.)"
        )
    return cls(**params)
