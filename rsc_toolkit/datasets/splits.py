"""Single source of the split file parser.

`official_split.txt` format: one `<recording_id> <split>` per line (whitespace-separated). Blank lines and `#` comments are skipped; split is lower-cased.
Shared by IndexDataset._load_official_split_map and respirenet_chunks.read_official_split.
"""
from __future__ import annotations

import pathlib
from typing import Dict, Union


def read_split_file(split_file: Union[str, pathlib.Path]) -> Dict[str, str]:
    """recording_id -> split ("train"/"test", etc., lower-case). FileNotFoundError if the file does not exist."""
    path = pathlib.Path(split_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    mapping: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        mapping[parts[0]] = parts[1].lower()
    return mapping


__all__ = ["read_split_file"]
