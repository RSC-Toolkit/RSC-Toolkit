"""튜토리얼이 기대는 진입점 — `python -m rsc_toolkit.cli`, toy 전용 캐시 디렉터리, 노트북 생성기."""
import pathlib
import subprocess
import sys

from hydra import compose, initialize_config_module

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_python_m_cli_shows_help_and_rejects_unknown_option():
    # 2026-09-14 이전에는 __main__ 가드가 없어 아무것도 안 하고 exit 0 이었다(toy 스크립트가 조용히 실패).
    out = subprocess.run([sys.executable, "-m", "rsc_toolkit.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "datasets" in out.stdout
    bad = subprocess.run([sys.executable, "-m", "rsc_toolkit.cli", "datasets", "build-index", "toy", "--forse"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "unrecognized" in bad.stderr      # parse_known_args 가 오타를 삼키던 문제


def test_toy_configs_use_their_own_cache_dir():
    # toy 는 ICBHI 파일·sample_id 형식을 그대로 쓰므로 공용 캐시를 쓰면 다른 config 의 피처를 조용히 읽는다 (REVIEW 2026-09-10 §7 P0).
    with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
        for data in ("toy", "toy_fbank"):
            cfg = compose(config_name="default", overrides=[f"data={data}", "callbacks=best_checkpoint", "run.group=g", "run.name=n"])
            assert cfg.data.params.features.cache_dir.endswith("features_cache_toy"), data
            assert cfg.data.params.data_dir == "downloaded_datasets/toy"
            assert cfg.trainer.params.callbacks[0].dirpath == "outputs/g/n"


def test_notebook_generator_matches_committed_notebook(tmp_path):
    import nbformat
    gen = ROOT / "scripts" / "tutorial" / "make_notebook.py"
    committed = nbformat.read(ROOT / "notebooks" / "tutorial.ipynb", as_version=4)
    # 생성기를 임시 위치로 실행해 커밋본과 셀 소스가 같은지 본다
    subprocess.run([sys.executable, str(gen), str(tmp_path / "nb.ipynb")], check=True, cwd=ROOT)
    fresh = nbformat.read(tmp_path / "nb.ipynb", as_version=4)
    assert [c.source for c in fresh.cells] == [c.source for c in committed.cells], "make_notebook.py 를 고쳤으면 다시 생성해 커밋하라"
