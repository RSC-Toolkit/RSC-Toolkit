"""
raw 데이터셋(다운로드·인덱스·CLI) 등록소 — 하위호환 창구.

등록의 원천은 `rsc_toolkit.registries.DATASETS` (공용 Registry) 다. 데이터셋 모듈은 데코레이터로 등록한다:

    from rsc_toolkit.datasets.factory import DatasetFactory
    @DatasetFactory.register_dataset_class("icbhi")
    class ICBHIDataset(BaseRSCDataset): ...

    from rsc_toolkit.registries import DATASETS   # 같은 것
    @DATASETS.register("heartsound")
    class HeartSoundDataset(BaseRSCDataset): ...

내장 4종(icbhi · sprsound · hflungv1 · toy)은 허브에 lazy 로 선언돼 있어 파일명 규약과 무관하게 항상 보인다.
패키지 안의 `*_dataset*` 모듈 스캔(register_datasets)은 예전 규약의 하위호환으로 유지하고, 외부 패키지는
엔트리포인트 `rsc_toolkit.plugins` / `RSC_TOOLKIT_PLUGINS` 로 붙는다 (rsc_toolkit.registry 참조).
같은 이름에 다른 클래스를 등록하면 ValueError — 예전의 침묵 덮어쓰기는 폐지했다.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Callable, List, Type

from rsc_toolkit.datasets.base import BaseRSCDataset
from rsc_toolkit.registries import DATASETS
from rsc_toolkit.registry import Registry, load_plugins

logger = logging.getLogger(__name__)

_SCANNED = False


class DatasetFactory:
    # Mapping (keys() 가 CLI choices) — 엔트리포인트 데이터셋도 자동으로 나타난다.
    datasets: Registry = DATASETS

    @staticmethod
    def register_datasets() -> None:
        """
        rsc_toolkit.datasets 아래 이름에 "_dataset" 이 들어간 모듈(하위 패키지 포함)을 import 해
        데코레이터가 실행되게 하고, 플러그인을 읽는다. 멱등. import 실패는 모듈 단위로 경고만 남긴다.
        """
        global _SCANNED
        if _SCANNED:
            return
        _SCANNED = True

        import rsc_toolkit.datasets as package

        def _try_import(name: str) -> None:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                logger.warning("데이터셋 모듈 %s 을 불러오지 못해 등록에서 제외한다 (선택 의존성 누락?): %s", name, exc)

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if "_dataset" in module_name:
                _try_import(f"rsc_toolkit.datasets.{module_name}")
                continue
            if not is_pkg:
                continue
            try:
                subpackage = importlib.import_module(f"rsc_toolkit.datasets.{module_name}")
            except ImportError as exc:
                logger.warning("데이터셋 하위 패키지 %s 을 불러오지 못했다: %s", module_name, exc)
                continue
            if not hasattr(subpackage, "__path__"):
                continue
            for _, submodule_name, _ in pkgutil.iter_modules(subpackage.__path__):
                if "_dataset" in submodule_name:
                    _try_import(f"rsc_toolkit.datasets.{module_name}.{submodule_name}")

        load_plugins()

    @classmethod
    def register_dataset_class(cls, dataset_name: str) -> Callable[[Type[BaseRSCDataset]], Type[BaseRSCDataset]]:
        """클래스 데코레이터. 같은 이름에 다른 클래스를 다시 등록하면 ValueError (같은 클래스는 멱등)."""
        return DATASETS.register(dataset_name)

    @classmethod
    def get_dataset_names(cls) -> List[str]:
        return list(DATASETS)

    @classmethod
    def get_dataset_class(cls, dataset_name: str) -> Type[BaseRSCDataset]:
        return DATASETS.get(dataset_name)  # 미지 이름: KeyError("Unknown dataset '...'. Available: [...]")
