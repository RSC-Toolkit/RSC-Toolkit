from typing import Any, Dict

import lightning as L

from rsc_toolkit.registries import DATAMODULES


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
    return cls(**params)
