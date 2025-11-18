from pathlib import Path

def fixture_path(name: str) -> Path:
    """
    Return absolute path to a fixture file in tests/fixtures/
    """
    return Path(__file__).resolve().parent.parent / "fixtures" / name
