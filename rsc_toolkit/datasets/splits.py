"""스플릿 파일 파서의 단일 출처 (P2-6).

`official_split.txt` 꼴: 줄마다 `<recording_id> <split>` (공백 구분). 빈 줄과 `#` 주석은 건너뛰고 split 은 소문자로.
IndexDataset._load_official_split_map 과 respirenet_chunks.read_official_split 이 각자 갖고 있던 파서를 합쳤다.
"""
from __future__ import annotations

import pathlib
from typing import Dict, Union


def read_split_file(split_file: Union[str, pathlib.Path]) -> Dict[str, str]:
    """recording_id -> split ("train"/"test" 등, 소문자). 파일이 없으면 FileNotFoundError."""
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
