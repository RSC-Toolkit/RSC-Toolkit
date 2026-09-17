"""Entry points the tutorial relies on — `python -m rsc_toolkit.cli`, the toy-only cache directory, the notebook generator."""
import pathlib
import subprocess
import sys

from hydra import compose, initialize_config_module

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_python_m_cli_shows_help_and_rejects_unknown_option():
    # Without a __main__ guard the module did nothing and exited 0 (the toy script failed silently).
    out = subprocess.run([sys.executable, "-m", "rsc_toolkit.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "datasets" in out.stdout
    bad = subprocess.run([sys.executable, "-m", "rsc_toolkit.cli", "datasets", "build-index", "toy", "--forse"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "unrecognized" in bad.stderr      # parse_known_args used to swallow typos


def test_toy_configs_use_their_own_cache_dir():
    # toy uses the ICBHI file and sample_id format as is, so a shared cache would silently read features from another config.
    with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
        for data in ("toy", "toy_fbank"):
            cfg = compose(config_name="default", overrides=[f"data={data}", "callbacks=best_checkpoint", "run.group=g", "run.name=n"])
            assert cfg.data.params.features.cache_dir.endswith("features_cache_toy"), data
            assert cfg.data.params.data_dir == "downloaded_datasets/toy"
            assert cfg.trainer.params.callbacks[0].dirpath == "outputs/g/n"


def test_notebook_generator_matches_committed_notebook(tmp_path):
    import pytest

    nbformat = pytest.importorskip("nbformat", reason="nbformat missing (notebook extra) — the CI unit-tests job installs it separately")
    gen = ROOT / "scripts" / "tutorial" / "make_notebook.py"
    committed = nbformat.read(ROOT / "notebooks" / "tutorial.ipynb", as_version=4)
    # run the generator into a temporary location and check that cell sources match the committed notebook
    subprocess.run([sys.executable, str(gen), str(tmp_path / "nb.ipynb")], check=True, cwd=ROOT)
    fresh = nbformat.read(tmp_path / "nb.ipynb", as_version=4)
    assert [c.source for c in fresh.cells] == [c.source for c in committed.cells], "make_notebook.py changed: regenerate the notebook and commit it"
