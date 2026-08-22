"""
패키지에 번들되는 데이터 파일이 실제로 배포되도록 package-data 글롭을 검사한다.

배경: pyproject 의 package-data 가 "metadata.csv" 로 적혀 있어 실제 위치인
data/metadata.csv 를 잡지 못했고, 휠에서 sprsound/hflungv1 의 번들 metadata.csv 가
빠졌다. editable 설치(-e)에서는 파일시스템을 그대로 보므로 드러나지 않았다.
휠을 직접 빌드하는 대신, 글롭이 실제 파일과 맞는지만 빠르게 본다.
"""
import pathlib

import pytest

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None

import rsc_toolkit


REPO_ROOT = pathlib.Path(rsc_toolkit.__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# 런타임 코드가 패키지 내부에서 읽는 파일들 — 반드시 어떤 글롭엔가 걸려야 한다.
RUNTIME_REQUIRED = [
    "rsc_toolkit/datasets/sprsound/data/metadata.csv",
    "rsc_toolkit/datasets/hflungv1/data/metadata.csv",
    "rsc_toolkit/datasets/icbhi/data/official_split.txt",
    "rsc_toolkit/datasets/icbhi/data/metadata.txt",
    "rsc_toolkit/configs/default.yaml",
]


def _package_data():
    if tomllib is None:
        pytest.skip("tomllib 은 Python 3.11+ 에서만 제공된다")
    with open(PYPROJECT, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["tool"]["setuptools"]["package-data"]


def _matched_files(package: str, pattern: str):
    pkg_dir = REPO_ROOT / package.replace(".", "/")
    return sorted(p for p in pkg_dir.glob(pattern) if p.is_file())


def test_every_package_data_entry_matches_something():
    """
    패키지 키마다 최소 한 글롭은 실제 파일을 잡아야 한다.

    글롭 하나하나를 강제하지는 않는다 — `configs/**/*.yml` 처럼 지금은 비어 있어도
    전방 호환용으로 둔 패턴이 있다. 키 전체가 빈 결과면 오타일 가능성이 높다
    (실제로 "rsc_toolkit.datasets.sprsound" = ["metadata.csv"] 가 그랬다).
    """
    for package, patterns in _package_data().items():
        matched = [f for pattern in patterns for f in _matched_files(package, pattern)]
        assert matched, f"package-data 항목이 아무 파일도 잡지 못한다: {package!r} / {patterns!r}"


@pytest.mark.parametrize("relpath", RUNTIME_REQUIRED)
def test_runtime_required_files_are_covered_by_package_data(relpath):
    """런타임이 읽는 번들 파일이 어떤 글롭에든 걸리는지 본다."""
    target = (REPO_ROOT / relpath).resolve()
    assert target.exists(), f"소스 트리에 없다: {relpath}"

    covered = any(
        target in _matched_files(package, pattern)
        for package, patterns in _package_data().items()
        for pattern in patterns
    )
    assert covered, f"package-data 글롭에 걸리지 않아 휠에서 빠진다: {relpath}"


def test_get_data_remnant_is_gone():
    """gdown 기반 잔재 스크립트가 패키지에 다시 들어오지 않는지 본다."""
    assert not (REPO_ROOT / "rsc_toolkit/datasets/sprsound/get_data.py").exists()
