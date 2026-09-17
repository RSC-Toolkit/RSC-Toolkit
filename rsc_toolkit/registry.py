"""Shared component registry.

Applies the three registration paths that ModelFactory established — package scan, entry points, and the
`RSC_TOOLKIT_PLUGINS` environment variable — uniformly to every component kind, not only models:
datamodule/loss/metric/optimizer/adapter/processor, etc.

Principles
- Registration uses a decorator (`@REG.register("name")`); built-in entries are only **declared** with
  `REG.register_lazy("name", "pkg.mod:Attr")` and imported on first lookup. If one entry's optional dependency is missing, only that entry fails.
- Registering a different object under an existing name raises ValueError. Re-registering the same object (module reload, etc.) is idempotent.
- Unknown names raise a KeyError listing the available names. `build(cfg)` is the single implementation of the
  `{"name": ..., "params": {...}}` convention and does not mutate cfg.
- Implements `Mapping`, so existing dict-consuming code (`name in REG`, `REG[name]`, `sorted(REG)`,
  argparse `choices=REG.keys()`) works unchanged. `pop`, used by tests, is supported too.

Plugins use a single entry-point group, `rsc_toolkit.plugins`. The older model-only group
`rsc_toolkit.models` is still read for backward compatibility.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import pkgutil
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Environment variable for attaching local modules without packaging (comma-separated). Same value as in models/factory.py.
PLUGIN_ENV_VAR = "RSC_TOOLKIT_PLUGINS"
# Unified entry-point group that registers components of every kind.
PLUGIN_ENTRY_POINT_GROUP = "rsc_toolkit.plugins"
# Backward compatibility: the older model-only group. Values are modules whose import runs the decorators.
LEGACY_ENTRY_POINT_GROUPS = ("rsc_toolkit.models",)

_PLUGINS_LOADED = False
_MISSING = object()


def load_plugins(*, force: bool = False) -> None:
    """Import plugin modules from entry points and the environment variable. By default once per process.

    A failing plugin does not affect the others (same principle as ModelFactory._register_plugins).
    `force=True` rescans even if already loaded — used by tests that re-register after changing the environment variable.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED and not force:
        return
    _PLUGINS_LOADED = True

    try:
        entry_points = importlib.metadata.entry_points()
    except Exception as exc:  # pragma: no cover - environment with broken metadata
        logger.warning("failed to read entry points: %s", exc)
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
                    logger.info("registered plugin: %s (entry point %s/%s)", entry_point.value, group, entry_point.name)
                except Exception as exc:
                    logger.warning(
                        "skipping plugin '%s' (%s): failed to load: %s", entry_point.name, entry_point.value, exc
                    )

    raw = os.environ.get(PLUGIN_ENV_VAR, "")
    for module_path in [part.strip() for part in raw.split(",") if part.strip()]:
        try:
            importlib.import_module(module_path)
            logger.info("registered plugin: %s (%s)", module_path, PLUGIN_ENV_VAR)
        except Exception as exc:
            logger.warning("%s: skipping module '%s': failed to load: %s", PLUGIN_ENV_VAR, module_path, exc)


def _qualified_name(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}:{getattr(obj, '__qualname__', getattr(obj, '__name__', '?'))}"


class Registry(Mapping):
    """Name -> class/callable registry. One instance per kind (see rsc_toolkit.registries)."""

    def __init__(
        self,
        kind: str,
        *,
        scan_packages: Sequence[str] = (),
        scan_exclude: Sequence[str] = (),
    ) -> None:
        self.kind = kind
        self._objs: Dict[str, Any] = {}
        self._lazy: Dict[str, str] = {}  # name -> "pkg.mod:Attr" (imported on first get)
        self._scan_packages = tuple(scan_packages)
        self._scan_exclude = set(scan_exclude)
        self._scanned = False
        self._failed: Dict[str, str] = {}  # modules that failed to import during the scan -> cause (appended to KeyError messages)

    # ── registration ────────────────────────────────────────────────────────
    def register(self, name: str) -> Callable[[Any], Any]:
        """Class/function decorator. Registering a different object under an existing name raises ValueError."""
        key = str(name).strip()
        if not key:
            raise ValueError(f"{self.kind} name is empty")

        def decorator(obj: Any) -> Any:
            existing = self._objs.get(key)
            if existing is not None and existing is not obj:
                raise ValueError(
                    f"{self.kind} name '{key}' is already registered to {_qualified_name(existing)}. "
                    f"Use a different name or remove the existing registration."
                )
            lazy_target = self._lazy.get(key)
            if lazy_target is not None and lazy_target != _qualified_name(obj):
                # A different object claims the name of a built-in lazy declaration. Compare path strings only,
                # without triggering an import (this may be the very module currently being imported).
                raise ValueError(
                    f"{self.kind} name '{key}' is declared as built-in entry {lazy_target}. "
                    f"Use a different name."
                )
            self._objs[key] = obj
            self._lazy.pop(key, None)
            return obj

        return decorator

    def register_lazy(self, name: str, target: str) -> None:
        """Declare a built-in entry. `target` is "pkg.module:Attr". The import happens on first lookup."""
        key = str(name).strip()
        if not key:
            raise ValueError(f"{self.kind} name is empty")
        if ":" not in target:
            raise ValueError(f"{self.kind} '{key}': lazy target must have the form 'pkg.module:Attr': {target!r}")
        if key in self._objs:
            raise ValueError(f"{self.kind} name '{key}' is already registered to {_qualified_name(self._objs[key])}")
        existing = self._lazy.get(key)
        if existing is not None and existing != target:
            raise ValueError(f"{self.kind} name '{key}' is already declared as {existing} (requested: {target})")
        self._lazy[key] = target

    def unregister(self, name: str) -> None:
        """Unregister a name. Unknown names raise KeyError. (For test cleanup; training code does not use it.)"""
        if name in self._objs:
            del self._objs[name]
        elif name in self._lazy:
            del self._lazy[name]
        else:
            raise KeyError(name)

    def pop(self, name: str, default: Any = _MISSING) -> Any:
        """dict.pop-compatible (tests clean up with `registry().pop(name, None)`)."""
        if name in self._objs:
            return self._objs.pop(name)
        if name in self._lazy:
            self._lazy.pop(name)
            return None
        if default is _MISSING:
            raise KeyError(name)
        return default

    # ── lookup ──────────────────────────────────────────────────────────────
    def get(self, name: str, default: Any = _MISSING) -> Any:  # type: ignore[override]
        """Look up an object by name. Unknown names raise a KeyError listing the available names (or return `default` if given)."""
        self._discover()
        if name in self._objs:
            return self._objs[name]
        if name in self._lazy:
            return self._resolve_lazy(name)
        if default is not _MISSING:
            return default
        hint = ""
        if self._failed:
            failed = "; ".join(f"{m}: {why}" for m, why in sorted(self._failed.items()))
            hint = f" Some modules failed to import during the scan and were not registered — {failed}"
        raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self)}.{hint}")

    @property
    def failed_modules(self) -> Dict[str, str]:
        """Modules that failed to import during the package scan, with causes (read-only copy for diagnostics)."""
        self._discover()
        return dict(self._failed)

    def _resolve_lazy(self, name: str) -> Any:
        target = self._lazy[name]
        module_path, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"{self.kind} '{name}' ({target}) is unavailable because a dependency is missing: {exc}. "
                f"Only this entry is affected; other {self.kind} entries still work."
            ) from exc
        try:
            obj = getattr(module, attr)
        except AttributeError as exc:
            raise ImportError(f"{self.kind} '{name}': {module_path} has no attribute {attr}") from exc
        self._objs[name] = obj
        self._lazy.pop(name, None)
        return obj

    def build(self, cfg: Mapping[str, Any], *args: Any, key: str = "name", **extra: Any) -> Any:
        """Single implementation of the `{"name": ..., "params": {...}}` convention. Copies params, so cfg is not mutated."""
        if not isinstance(cfg, Mapping):
            raise TypeError(f"{self.kind} config must be a mapping: {type(cfg).__name__}")
        name = cfg.get(key)
        if name is None:
            raise KeyError(f"{self.kind} config has no '{key}'. Got keys: {sorted(cfg)}")
        params = dict(cfg.get("params") or {})
        params.update(extra)
        return self.get(name)(*args, **params)

    # ── discovery: package scan (once) + plugins (once per process) ─────────
    def _discover(self) -> None:
        if not self._scanned:
            # Mark first to prevent recursion when the imports below touch this registry again.
            self._scanned = True
            try:
                for package_name in self._scan_packages:
                    self._scan_package(package_name)
            except BaseException:
                # If the scan is interrupted, the "done" flag would leave later modules permanently unregistered — reset it so the next lookup rescans.
                self._scanned = False
                raise
        # Plugins load after the built-in scan, so on a name conflict the plugin fails rather than the built-in entry.
        load_plugins()

    def _scan_package(self, package_name: str) -> None:
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            logger.warning("%s: failed to import scan package %s: %s", self.kind, package_name, exc)
            return
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg or module_name.startswith("_") or module_name in self._scan_exclude:
                continue
            full = f"{package_name}.{module_name}"
            try:
                importlib.import_module(full)
            except Exception as exc:
                # Catch every exception, not only ImportError: an internal error (NameError, AttributeError, a
                # ValueError from a name conflict) would otherwise abort the whole scan and drop all later modules.
                # Isolate per module and record the failure so the KeyError can point to the cause.
                self._failed[full] = f"{type(exc).__name__}: {exc}"
                level = logging.WARNING if isinstance(exc, ImportError) else logging.ERROR
                logger.log(
                    level,
                    "%s: module %s failed to import and is excluded from registration (%s): %s",
                    self.kind, full, "missing optional dependency?" if isinstance(exc, ImportError) else "error inside the module", exc,
                )

    def rescan(self) -> None:
        """Redo the package scan and plugin loading (idempotent; for tests and the REPL)."""
        self._scanned = False
        self._failed.clear()
        self._discover()
        load_plugins(force=True)

    # ── Mapping protocol ────────────────────────────────────────────────────
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
