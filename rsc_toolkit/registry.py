"""공용 컴포넌트 등록소(Registry).

ModelFactory 가 검증해 온 세 가지 등록 경로 — 패키지 스캔 · 엔트리포인트 · `RSC_TOOLKIT_PLUGINS`
환경변수 — 를 모델뿐 아니라 datamodule/loss/metric/optimizer/adapter/processor 등 모든 컴포넌트
종류에 같은 방식으로 적용한다. (2026-09-01 심층 리뷰의 통합 설계 §3-1, 리팩토링 P1 1단계.)

원칙
- 등록은 데코레이터(`@REG.register("name")`), 내장 항목은 `REG.register_lazy("name", "pkg.mod:Attr")`
  로 **선언만** 하고 import 는 첫 조회 때 한다. 한 항목의 선택 의존성이 빠져도 그 항목만 실패한다.
- 같은 이름에 다른 객체를 다시 등록하면 ValueError. 같은 객체(모듈 재로드 등)는 멱등.
- 미지 이름은 가용 목록을 붙인 KeyError. `build(cfg)` 는 `{"name": ..., "params": {...}}` 규약의
  단일 구현이고 cfg 를 파괴하지 않는다.
- `Mapping` 을 구현하므로 기존 dict 소비 코드(`name in REG`, `REG[name]`, `sorted(REG)`,
  argparse `choices=REG.keys()`)가 무수정으로 동작한다. 테스트가 쓰는 `pop` 도 지원한다.

플러그인은 엔트리포인트 그룹 `rsc_toolkit.plugins` 하나로 통일한다. 기존 모델 전용 그룹
`rsc_toolkit.models` 는 하위호환으로 계속 읽는다.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import pkgutil
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# 패키징 없이 로컬 모듈을 붙일 때 쓰는 환경변수 (쉼표 구분). models/factory.py 와 같은 값.
PLUGIN_ENV_VAR = "RSC_TOOLKIT_PLUGINS"
# 모든 종류의 컴포넌트를 등록하는 통합 엔트리포인트 그룹.
PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.plugins"
# 하위호환: 예전 모델 전용 그룹. 값은 import 하면 데코레이터가 실행되는 모듈이다.
LEGACY_ENTRY_POINT_GROUPS = ("rsc_toolkit.models",)

_PLUGINS_LOADED = False
_MISSING = object()


def load_plugins(*, force: bool = False) -> None:
    """엔트리포인트·환경변수 플러그인 모듈을 import 한다. 기본은 프로세스당 1회.

    한 플러그인이 실패해도 나머지는 살린다 (ModelFactory._register_plugins 의 원칙 유지).
    `force=True` 면 이미 로드했어도 다시 훑는다 — 테스트가 환경변수를 바꾼 뒤 재등록할 때 쓴다.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED and not force:
        return
    _PLUGINS_LOADED = True

    try:
        entry_points = importlib.metadata.entry_points()
    except Exception as exc:  # pragma: no cover - 메타데이터가 깨진 환경
        logger.warning("엔트리포인트 목록을 읽지 못했다: %s", exc)
        entry_points = None

    if entry_points is not None:
        for group in (PLUGIN_ENTRY_POINT_GROUP, *LEGACY_ENTRY_POINT_GROUPS):
            if hasattr(entry_points, "select"):
                found = list(entry_points.select(group=group))
            else:  # pragma: no cover - Python 3.9
                found = list(entry_points.get(group, []))
            for entry_point in found:
                try:
                    entry_point.load()
                    logger.info("플러그인 등록: %s (엔트리포인트 %s/%s)", entry_point.value, group, entry_point.name)
                except Exception as exc:
                    logger.warning(
                        "플러그인 '%s' (%s) 을 불러오지 못해 건너뛴다: %s", entry_point.name, entry_point.value, exc
                    )

    raw = os.environ.get(PLUGIN_ENV_VAR, "")
    for module_path in [part.strip() for part in raw.split(",") if part.strip()]:
        try:
            importlib.import_module(module_path)
            logger.info("플러그인 등록: %s (%s)", module_path, PLUGIN_ENV_VAR)
        except Exception as exc:
            logger.warning("%s 가 가리키는 모듈 '%s' 을 불러오지 못해 건너뛴다: %s", PLUGIN_ENV_VAR, module_path, exc)


def _qualified_name(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}:{getattr(obj, '__qualname__', getattr(obj, '__name__', '?'))}"


class Registry(Mapping):
    """이름 -> 클래스/호출가능 등록소. 종류(kind)마다 하나씩 만든다 (rsc_toolkit.registries 참조)."""

    def __init__(
        self,
        kind: str,
        *,
        scan_packages: Sequence[str] = (),
        scan_exclude: Sequence[str] = (),
    ) -> None:
        self.kind = kind
        self._objs: Dict[str, Any] = {}
        self._lazy: Dict[str, str] = {}  # name -> "pkg.mod:Attr" (첫 get 때 import)
        self._scan_packages = tuple(scan_packages)
        self._scan_exclude = set(scan_exclude)
        self._scanned = False

    # ── 등록 ────────────────────────────────────────────────────────────────
    def register(self, name: str) -> Callable[[Any], Any]:
        """클래스/함수 데코레이터. 같은 이름에 다른 객체를 다시 등록하면 ValueError."""
        key = str(name).strip()
        if not key:
            raise ValueError(f"{self.kind} 이름이 비어 있다")

        def decorator(obj: Any) -> Any:
            existing = self._objs.get(key)
            if existing is not None and existing is not obj:
                raise ValueError(
                    f"{self.kind} 이름 '{key}' 은 이미 {_qualified_name(existing)} 에 등록돼 있다. "
                    f"다른 이름을 쓰거나 기존 등록을 정리하라."
                )
            lazy_target = self._lazy.get(key)
            if lazy_target is not None and lazy_target != _qualified_name(obj):
                # 내장 lazy 선언과 다른 객체가 같은 이름을 요구한다. import 를 유발하지 않고
                # (자기 자신을 import 중인 모듈일 수 있다) 경로 문자열로만 비교한다.
                raise ValueError(
                    f"{self.kind} 이름 '{key}' 은 내장 항목 {lazy_target} 으로 선언돼 있다. "
                    f"다른 이름을 쓰라."
                )
            self._objs[key] = obj
            self._lazy.pop(key, None)
            return obj

        return decorator

    def register_lazy(self, name: str, target: str) -> None:
        """내장 항목 선언. `target` 은 "pkg.module:Attr". import 는 첫 조회 때 일어난다."""
        key = str(name).strip()
        if not key:
            raise ValueError(f"{self.kind} 이름이 비어 있다")
        if ":" not in target:
            raise ValueError(f"{self.kind} '{key}' 의 lazy 대상은 'pkg.module:Attr' 꼴이어야 한다: {target!r}")
        if key in self._objs:
            raise ValueError(f"{self.kind} 이름 '{key}' 은 이미 {_qualified_name(self._objs[key])} 에 등록돼 있다")
        existing = self._lazy.get(key)
        if existing is not None and existing != target:
            raise ValueError(f"{self.kind} 이름 '{key}' 은 이미 {existing} 으로 선언돼 있다 (요청: {target})")
        self._lazy[key] = target

    def unregister(self, name: str) -> None:
        """등록 해제. 없는 이름은 KeyError. (테스트 정리용 — 학습 코드는 쓰지 않는다.)"""
        if name in self._objs:
            del self._objs[name]
        elif name in self._lazy:
            del self._lazy[name]
        else:
            raise KeyError(name)

    def pop(self, name: str, default: Any = _MISSING) -> Any:
        """dict.pop 호환 (기존 테스트가 `registry().pop(name, None)` 으로 정리한다)."""
        if name in self._objs:
            return self._objs.pop(name)
        if name in self._lazy:
            self._lazy.pop(name)
            return None
        if default is _MISSING:
            raise KeyError(name)
        return default

    # ── 조회 ────────────────────────────────────────────────────────────────
    def get(self, name: str, default: Any = _MISSING) -> Any:  # type: ignore[override]
        """이름으로 객체를 찾는다. 미지 이름은 가용 목록을 붙인 KeyError (default 를 주면 그것을 반환)."""
        self._discover()
        if name in self._objs:
            return self._objs[name]
        if name in self._lazy:
            return self._resolve_lazy(name)
        if default is not _MISSING:
            return default
        raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self)}")

    def _resolve_lazy(self, name: str) -> Any:
        target = self._lazy[name]
        module_path, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"{self.kind} '{name}' ({target}) 의 의존성이 없어 쓸 수 없다: {exc}. "
                f"이 항목만의 문제이며 다른 {self.kind} 는 영향이 없다."
            ) from exc
        try:
            obj = getattr(module, attr)
        except AttributeError as exc:
            raise ImportError(f"{self.kind} '{name}': {module_path} 에 {attr} 가 없다") from exc
        self._objs[name] = obj
        self._lazy.pop(name, None)
        return obj

    def build(self, cfg: Mapping[str, Any], *args: Any, key: str = "name", **extra: Any) -> Any:
        """`{"name": ..., "params": {...}}` 규약의 단일 구현. params 를 복사하므로 cfg 를 파괴하지 않는다."""
        if not isinstance(cfg, Mapping):
            raise TypeError(f"{self.kind} config 는 매핑이어야 한다: {type(cfg).__name__}")
        name = cfg.get(key)
        if name is None:
            raise KeyError(f"{self.kind} config 에 '{key}' 가 없다. 받은 키: {sorted(cfg)}")
        params = dict(cfg.get("params") or {})
        params.update(extra)
        return self.get(name)(*args, **params)

    # ── 자동 발견: 패키지 스캔(1회) + 플러그인(프로세스 1회) ────────────────
    def _discover(self) -> None:
        if not self._scanned:
            # 아래 import 가 다시 이 registry 를 건드리는 재귀를 막기 위해 먼저 표시한다.
            self._scanned = True
            for package_name in self._scan_packages:
                self._scan_package(package_name)
        load_plugins()

    def _scan_package(self, package_name: str) -> None:
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            logger.warning("%s 스캔 패키지 %s 를 불러오지 못했다: %s", self.kind, package_name, exc)
            return
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg or module_name.startswith("_") or module_name in self._scan_exclude:
                continue
            try:
                importlib.import_module(f"{package_name}.{module_name}")
            except ImportError as exc:
                logger.warning(
                    "%s 모듈 %s.%s 을 불러오지 못해 등록에서 제외한다 (선택 의존성 누락?): %s",
                    self.kind, package_name, module_name, exc,
                )

    def rescan(self) -> None:
        """패키지 스캔과 플러그인 로드를 다시 한다 (멱등; 테스트·REPL 용)."""
        self._scanned = False
        self._discover()
        load_plugins(force=True)

    # ── Mapping 프로토콜 ────────────────────────────────────────────────────
    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        self._discover()
        return name in self._objs or name in self._lazy

    def __iter__(self) -> Iterator[str]:
        self._discover()
        return iter(sorted({*self._objs, *self._lazy}))

    def __len__(self) -> int:
        self._discover()
        return len({*self._objs, *self._lazy})

    def __delitem__(self, name: str) -> None:
        self.unregister(name)

    def names(self) -> list[str]:
        return list(self)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self)} entries)"


__all__ = [
    "Registry",
    "load_plugins",
    "PLUGIN_ENV_VAR",
    "PLUGIN_ENTRY_POINT_GROUP",
    "LEGACY_ENTRY_POINT_GROUPS",
]
