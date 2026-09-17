"""Contract of the shared Registry.

Pinned here: rejection of duplicate registration and idempotent re-registration, lazy declarations and import on first lookup, ImportError isolation,
KeyError listing the available names for an unknown name, non-mutating params in build(), the Mapping protocol (compatibility with dict-consuming code),
idempotence and isolation of the plugin loader.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from rsc_toolkit import registry as registry_mod
from rsc_toolkit.registry import PLUGIN_ENV_VAR, Registry, load_plugins


class TestRegisterSemantics:
    def test_decorator_registers_and_returns_object(self):
        reg = Registry("thing")

        @reg.register("a")
        class A:
            pass

        assert reg.get("a") is A
        assert reg["a"] is A
        assert "a" in reg and len(reg) == 1 and list(reg) == ["a"]

    def test_same_object_reregistration_is_idempotent(self):
        reg = Registry("thing")

        class A:
            pass

        reg.register("a")(A)
        reg.register("a")(A)  # module reload etc. — passes silently
        assert reg.get("a") is A

    def test_different_object_same_name_is_rejected(self):
        reg = Registry("thing")

        @reg.register("a")
        class A:
            pass

        with pytest.raises(ValueError, match="'a'"):

            @reg.register("a")
            class B:
                pass

        assert reg.get("a") is A

    def test_empty_name_is_rejected(self):
        reg = Registry("thing")
        with pytest.raises(ValueError):
            reg.register("  ")
        with pytest.raises(ValueError):
            reg.register_lazy("", "os:path")

    def test_unregister_and_pop(self):
        reg = Registry("thing")
        reg.register("a")(object)
        reg.register_lazy("b", "os.path:join")
        assert reg.pop("a") is object
        assert reg.pop("b") is None  # lazy entries are dropped without resolving
        assert reg.pop("zzz", None) is None
        with pytest.raises(KeyError):
            reg.pop("zzz")
        with pytest.raises(KeyError):
            reg.unregister("zzz")
        assert len(reg) == 0


class TestLazy:
    def test_lazy_is_resolved_on_first_get_only(self):
        reg = Registry("thing")
        reg.register_lazy("join", "os.path:join")
        assert "join" in reg  # existence is known without lookup
        assert reg._objs == {}  # not imported yet
        import os.path

        assert reg.get("join") is os.path.join
        assert "join" in reg._objs and "join" not in reg._lazy

    def test_lazy_target_shape_is_validated(self):
        reg = Registry("thing")
        with pytest.raises(ValueError, match="pkg.module:Attr"):
            reg.register_lazy("x", "os.path.join")

    def test_lazy_duplicate_declaration(self):
        reg = Registry("thing")
        reg.register_lazy("x", "os.path:join")
        reg.register_lazy("x", "os.path:join")  # same target is idempotent
        with pytest.raises(ValueError):
            reg.register_lazy("x", "os.path:split")

    def test_decorator_matching_lazy_declaration_is_accepted(self):
        """Registering a built-in entry with a decorator inside its own module (same path) must also pass."""
        reg = Registry("thing")
        reg.register_lazy("R", f"{__name__}:_Dummy")
        assert reg.register("R")(_Dummy) is _Dummy
        assert "R" not in reg._lazy and reg.get("R") is _Dummy

    def test_decorator_conflicting_with_lazy_declaration_is_rejected(self):
        reg = Registry("thing")
        reg.register_lazy("R", "os.path:join")
        with pytest.raises(ValueError, match="built-in entry"):
            reg.register("R")(_Dummy)

    def test_missing_dependency_only_breaks_that_entry(self):
        reg = Registry("thing")
        reg.register_lazy("broken", "rsc_definitely_missing_pkg.mod:Cls")
        reg.register_lazy("fine", "os.path:join")
        with pytest.raises(ImportError, match="'broken'"):
            reg.get("broken")
        import os.path

        assert reg.get("fine") is os.path.join
        assert "broken" in reg  # the declaration remains; retrying gives the same error

    def test_missing_attribute_is_import_error(self):
        reg = Registry("thing")
        reg.register_lazy("x", "os.path:no_such_attr_xyz")
        with pytest.raises(ImportError, match="no_such_attr_xyz"):
            reg.get("x")


class TestLookupAndBuild:
    def test_unknown_name_lists_available(self):
        reg = Registry("widget")
        reg.register("a")(object)
        with pytest.raises(KeyError, match=r"Unknown widget 'zzz'. Available: \['a'\]"):
            reg.get("zzz")
        with pytest.raises(KeyError, match="Available"):
            reg["zzz"]
        assert reg.get("zzz", None) is None

    def test_build_copies_params_and_passes_positional_and_extra(self):
        reg = Registry("thing")
        calls = []

        @reg.register("rec")
        def make(*args, **kwargs):
            calls.append((args, kwargs))
            return "built"

        cfg = {"name": "rec", "params": {"lr": 1e-3}}
        assert reg.build(cfg, "pos", extra=True) == "built"
        assert calls == [(("pos",), {"lr": 1e-3, "extra": True})]
        assert cfg == {"name": "rec", "params": {"lr": 1e-3}}  # not mutated

    def test_build_without_name_or_with_none_params(self):
        reg = Registry("thing")
        reg.register("rec")(lambda **kw: kw)
        with pytest.raises(KeyError, match="has no 'name'"):
            reg.build({"params": {}})
        assert reg.build({"name": "rec", "params": None}) == {}
        assert reg.build({"name": "rec"}) == {}
        with pytest.raises(TypeError):
            reg.build(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_build_custom_key(self):
        reg = Registry("thing")
        reg.register("rec")(lambda **kw: kw)
        assert reg.build({"kind": "rec", "params": {"a": 1}}, key="kind") == {"a": 1}


class TestMappingProtocol:
    def test_dict_consumer_idioms(self):
        reg = Registry("thing")
        reg.register("b")(int)
        reg.register_lazy("a", "os.path:join")
        assert sorted(reg) == ["a", "b"]
        assert list(reg.keys()) == ["a", "b"]
        assert dict(reg)["b"] is int  # values() resolves lazy entries
        assert set(reg.items()) >= {("b", int)}
        assert len(reg) == 2
        del reg["a"]
        assert "a" not in reg
        assert repr(reg) == "Registry('thing', 1 entries)"


class TestPackageScan:
    def test_scan_imports_modules_once_and_tolerates_import_errors(self, tmp_path, monkeypatch):
        pkg = tmp_path / "rsc_scan_fixture_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "good.py").write_text(
            textwrap.dedent(
                """
                from rsc_toolkit.registry import Registry
                import rsc_scan_fixture_pkg as p
                @p.REG.register("good")
                class Good: pass
                """
            )
        )
        (pkg / "bad.py").write_text("import rsc_definitely_missing_pkg\n")
        (pkg / "_private.py").write_text("raise RuntimeError('must not be imported')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            import rsc_scan_fixture_pkg as p

            p.REG = Registry("scan", scan_packages=("rsc_scan_fixture_pkg",))
            assert "good" in p.REG  # the scan imported good.py
            assert p.REG._scanned is True
            names_before = list(p.REG)
            p.REG._discover()  # idempotent
            assert list(p.REG) == names_before
        finally:
            for m in list(sys.modules):
                if m.startswith("rsc_scan_fixture_pkg"):
                    sys.modules.pop(m)


    def test_internal_module_error_is_isolated_and_reported(self, tmp_path, monkeypatch):
        """Internal errors other than ImportError (NameError etc.) are also isolated per module, and the KeyError message names the cause
        (previously the whole scan aborted and modules alphabetically after it stayed unregistered permanently)."""
        pkg = tmp_path / "rsc_scan_fixture_pkg2"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "a_broken.py").write_text("undefined_name_here\n")          # NameError (alphabetically first module)
        (pkg / "b_good.py").write_text(
            textwrap.dedent(
                """
                import rsc_scan_fixture_pkg2 as p
                @p.REG.register("good2")
                class Good2: pass
                """
            )
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            import rsc_scan_fixture_pkg2 as p

            p.REG = Registry("scan2", scan_packages=("rsc_scan_fixture_pkg2",))
            assert "good2" in p.REG                                    # the later module survived
            assert list(p.REG.failed_modules) == ["rsc_scan_fixture_pkg2.a_broken"]
            assert p.REG.failed_modules["rsc_scan_fixture_pkg2.a_broken"].startswith("NameError")
            with pytest.raises(KeyError) as exc:
                p.REG.get("nope")
            assert "a_broken" in str(exc.value) and "NameError" in str(exc.value)
            p.REG.rescan()
            assert "good2" in p.REG and "rsc_scan_fixture_pkg2.a_broken" in p.REG.failed_modules
        finally:
            for m in list(sys.modules):
                if m.startswith("rsc_scan_fixture_pkg2"):
                    sys.modules.pop(m)


class TestPluginLoader:
    def test_env_var_plugin_is_imported_and_broken_one_is_skipped(self, tmp_path, monkeypatch):
        module_name = "rsc_registry_env_plugin_fixture"
        (tmp_path / f"{module_name}.py").write_text("LOADED = True\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv(PLUGIN_ENV_VAR, f"{module_name}, rsc_definitely_missing_plugin")
        try:
            load_plugins(force=True)  # does not raise even with a broken module
            assert sys.modules[module_name].LOADED is True
        finally:
            sys.modules.pop(module_name, None)

    def test_load_plugins_is_once_per_process_unless_forced(self, monkeypatch):
        monkeypatch.setattr(registry_mod, "_PLUGINS_LOADED", False)
        monkeypatch.setenv(PLUGIN_ENV_VAR, "")
        load_plugins()
        assert registry_mod._PLUGINS_LOADED is True
        monkeypatch.setenv(PLUGIN_ENV_VAR, "rsc_definitely_missing_plugin")
        load_plugins()  # second call is a no-op (does not retry the broken module)
        load_plugins(force=True)  # force rescans — still without raising


class _Dummy:
    pass
