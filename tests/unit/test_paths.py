"""rsc_toolkit.paths: resolution rules for RSCTK_PRETRAINED_ROOT / RSCTK_PROJECT_ROOT."""
from __future__ import annotations

import os
import pathlib

import pytest

from rsc_toolkit import paths


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(paths.PRETRAINED_ROOT_ENV, raising=False)
    monkeypatch.delenv(paths.PROJECT_ROOT_ENV, raising=False)


def test_resolve_pretrained_without_root_is_cwd_relative(tmp_path, monkeypatch, no_env):
    monkeypatch.chdir(tmp_path)
    cands = ("pretrained_models/upstream/a.pth", "pretrained_models/a.pth")
    assert paths.resolve_pretrained(cands) is None
    (tmp_path / "pretrained_models").mkdir()
    (tmp_path / "pretrained_models" / "a.pth").write_bytes(b"x")
    assert paths.resolve_pretrained(cands) == pathlib.Path("pretrained_models/a.pth")
    assert paths.download_target(cands[0]) == pathlib.Path(cands[0])


def test_resolve_pretrained_prefers_env_root(tmp_path, monkeypatch, no_env):
    root = tmp_path / "shared"
    (root / "upstream").mkdir(parents=True)
    (root / "upstream" / "a.pth").write_bytes(b"x")
    (root / "b.pth").write_bytes(b"x")
    monkeypatch.setenv(paths.PRETRAINED_ROOT_ENV, str(root))
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_pretrained(("pretrained_models/upstream/a.pth",)) == root / "upstream" / "a.pth"
    assert paths.resolve_pretrained(("pretrained_models/upstream/dass/b.pth",)) == root / "b.pth"  # basename fallback
    assert paths.download_target("pretrained_models/upstream/new.pth") == root / "upstream" / "new.pth"
    # explicit path: used as is if it exists; if relative and missing, looked up under ROOT; otherwise returned unchanged (for the caller's error message)
    assert paths.resolve_pretrained(explicit=str(root / "b.pth")) == root / "b.pth"
    assert paths.resolve_pretrained(explicit="pretrained_models/upstream/a.pth") == root / "upstream" / "a.pth"
    assert paths.resolve_pretrained(explicit="pretrained_models/nope.pth") == pathlib.Path("pretrained_models/nope.pth")
    assert paths.resolve_pretrained(explicit="/abs/nope.pth") == pathlib.Path("/abs/nope.pth")


def test_project_root_from_order(tmp_path, monkeypatch, no_env):
    explicit = tmp_path / "x"
    explicit.mkdir()
    assert paths.project_root_from(explicit) == explicit.resolve()
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    assert paths.project_root_from(None) == tmp_path.resolve()
    monkeypatch.delenv(paths.PROJECT_ROOT_ENV)
    work = tmp_path / "work"
    work.mkdir()
    (work / "pyproject.toml").write_text("")
    monkeypatch.chdir(work)
    assert paths.project_root_from(None) == work.resolve()  # cwd if it has repository markers
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(bare)
    r = paths.project_root_from(None)  # cwd without markers -> dev checkout (pyproject above the package) or cwd
    assert r == bare.resolve() or (r / "pyproject.toml").exists()


def test_model_resolvers_honor_env_root(tmp_path, monkeypatch, no_env):
    """Per-model candidate-list resolvers use the shared rule — they find files under ROOT."""
    from rsc_toolkit.models import ast_amsd, beats_pafa

    root = tmp_path / "w"
    (root / "upstream").mkdir(parents=True)
    (root / "upstream" / "BEATs_iter3_plus_AS2M.pt").write_bytes(b"x")
    (root / "upstream" / "audioset_16_16_0.4422.pth").write_bytes(b"x")
    monkeypatch.setenv(paths.PRETRAINED_ROOT_ENV, str(root))
    monkeypatch.chdir(tmp_path)
    assert beats_pafa._resolve_beats_ckpt(None) == root / "upstream" / "BEATs_iter3_plus_AS2M.pt"
    assert ast_amsd._resolve_ckpt(None) == root / "upstream" / "audioset_16_16_0.4422.pth"
    monkeypatch.delenv(paths.PRETRAINED_ROOT_ENV)
    with pytest.raises(FileNotFoundError):
        beats_pafa._resolve_beats_ckpt(None)
