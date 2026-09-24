"""
Checks the package-data globs so that data files bundled in the package are actually shipped.

Background: package-data in pyproject listed "metadata.csv", which did not match the actual location
data/metadata.csv, so the bundled metadata.csv of sprsound/hflungv1 was missing from the wheel.
An editable install (-e) reads the filesystem directly, so it did not show up there.
Instead of building a wheel, this quickly checks that the globs match real files.
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

# Files runtime code reads from inside the package — each must be matched by some glob.
RUNTIME_REQUIRED = [
    "rsc_toolkit/datasets/sprsound/data/metadata.csv",
    "rsc_toolkit/datasets/hflungv1/data/metadata.csv",
    "rsc_toolkit/datasets/icbhi/data/official_split.txt",
    "rsc_toolkit/datasets/icbhi/data/metadata.txt",
    "rsc_toolkit/configs/default.yaml",
]


def _package_data():
    if tomllib is None:
        pytest.skip("tomllib is only available in Python 3.11+")
    with open(PYPROJECT, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["tool"]["setuptools"]["package-data"]


def _matched_files(package: str, pattern: str):
    pkg_dir = REPO_ROOT / package.replace(".", "/")
    return sorted(p for p in pkg_dir.glob(pattern) if p.is_file())


def test_every_package_data_entry_matches_something():
    """
    For each package key, at least one glob must match a real file.

    Not every single glob is enforced — some patterns such as `configs/**/*.yml` are kept for
    forward compatibility even though nothing matches yet. An empty result for a whole key is most likely a typo
    (as actually happened with "rsc_toolkit.datasets.sprsound" = ["metadata.csv"]).
    """
    for package, patterns in _package_data().items():
        matched = [f for pattern in patterns for f in _matched_files(package, pattern)]
        assert matched, f"package-data entry matches no file: {package!r} / {patterns!r}"


@pytest.mark.parametrize("relpath", RUNTIME_REQUIRED)
def test_runtime_required_files_are_covered_by_package_data(relpath):
    """Bundled files read at runtime are matched by some glob."""
    target = (REPO_ROOT / relpath).resolve()
    assert target.exists(), f"not in the source tree: {relpath}"

    covered = any(
        target in _matched_files(package, pattern)
        for package, patterns in _package_data().items()
        for pattern in patterns
    )
    assert covered, f"not matched by any package-data glob, so missing from the wheel: {relpath}"


def test_get_data_remnant_is_gone():
    """The leftover gdown-based script does not come back into the package."""
    assert not (REPO_ROOT / "rsc_toolkit/datasets/sprsound/get_data.py").exists()
