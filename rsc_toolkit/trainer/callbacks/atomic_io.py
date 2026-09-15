"""best.pth / best.json / final_state.pth 의 원자적 저장.

예전엔 torch.save·json.dump 가 대상 파일을 직접 덮어썼다 — 저장 도중(디스크 풀·프로세스 종료) 죽으면 이전 best 가
손상된 채 남아 aggregate·재검증이 그 run 을 통째로 잃는다 (REVIEW 2026-09-10 §4 P2). last.ckpt 가 이미 쓰던
tmp + os.replace 규약을 그대로 옮긴다: 같은 디렉터리에 `<이름>.tmp.<pid>` 로 쓴 뒤 교체하므로 어느 순간에도
대상 경로에는 온전한 이전 파일 또는 온전한 새 파일만 있다.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import torch


def _tmp_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def atomic_torch_save(obj: Any, path: os.PathLike | str) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_json_dump(obj: Any, path: os.PathLike | str, *, indent: int = 1, **kwargs: Any) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_text(json.dumps(obj, indent=indent, **kwargs))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
