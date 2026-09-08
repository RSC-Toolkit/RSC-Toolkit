"""경로 해석의 단일 출처 (리팩토링 P2-9) — 저장소 밖에서 설치해 써도 되게.

두 가지를 한 곳에서 정한다:

1. **사전학습 체크포인트** (`pretrained_models/upstream/…`). 모델마다 후보 목록을 cwd 기준으로 찾고 없으면 내려받았는데,
   이제 `RSCTK_PRETRAINED_ROOT` 환경변수를 먼저 본다 — 여러 checkout 이 한 디렉터리의 가중치를 공유하거나, 패키지로
   설치한 사용자가 가중치를 아무 데나 둘 수 있게. 해석 순서:
     명시 경로(config 의 pretrained_path)              → 그대로(상대면 cwd, 없으면 ROOT 아래에서도 찾는다)
     각 후보 경로 c 에 대해                              → ROOT/<c 에서 'pretrained_models/' 접두사를 뗀 것> → ROOT/<basename> → cwd/c
   내려받을 때의 목적지도 같은 규칙(ROOT 가 있으면 ROOT 아래).

2. **프로젝트 루트**. 몇몇 모듈이 `Path(__file__).parents[2]` 로 저장소 루트를 추정했는데 설치본에서는 site-packages 를
   가리킨다. `project_root_from()` 은 인자 → `RSCTK_PROJECT_ROOT` → cwd(저장소 표지가 있으면) → 패키지 상위(개발 checkout) → cwd.
"""
from __future__ import annotations

import os
import pathlib
from typing import Iterable, Optional, Sequence, Union

PRETRAINED_ROOT_ENV = "RSCTK_PRETRAINED_ROOT"
PROJECT_ROOT_ENV = "RSCTK_PROJECT_ROOT"
_PRETRAINED_PREFIX = "pretrained_models/"
_ROOT_MARKERS = ("pyproject.toml", ".git", "rsc_toolkit", ".datasets", "pretrained_models")

PathLike = Union[str, os.PathLike]


def pretrained_root() -> Optional[pathlib.Path]:
    """`RSCTK_PRETRAINED_ROOT` (없으면 None)."""
    raw = os.environ.get(PRETRAINED_ROOT_ENV, "").strip()
    return pathlib.Path(raw).expanduser() if raw else None


def _under_root(root: pathlib.Path, candidate: str) -> Iterable[pathlib.Path]:
    rel = candidate[len(_PRETRAINED_PREFIX):] if candidate.startswith(_PRETRAINED_PREFIX) else candidate
    yield root / rel
    yield root / pathlib.Path(candidate).name


def resolve_pretrained(candidates: Sequence[str] = (), explicit: Optional[PathLike] = None) -> Optional[pathlib.Path]:
    """존재하는 체크포인트 경로를 돌려준다 (없으면 None — 호출자가 내려받거나 오류를 낸다).

    `explicit` 이 있으면 그것을 우선한다: 절대/존재하는 상대 경로면 그대로, 상대인데 없으면 ROOT 아래에서도 찾고,
    그래도 없으면 **그 경로를 그대로** 돌려준다(호출자의 기존 오류 메시지가 그 경로를 지목하도록).
    """
    root = pretrained_root()
    if explicit is not None:
        p = pathlib.Path(explicit).expanduser()
        if p.exists() or p.is_absolute():
            return p
        if root is not None:
            for q in _under_root(root, str(p)):
                if q.exists():
                    return q
        return p
    for cand in candidates:
        if root is not None:
            for q in _under_root(root, cand):
                if q.exists():
                    return q
        p = pathlib.Path(cand).expanduser()
        if p.exists():
            return p
    return None


def download_target(candidate: str) -> pathlib.Path:
    """체크포인트를 내려받을 자리 — ROOT 가 있으면 ROOT 아래(접두사 제거), 없으면 cwd 기준 후보 경로 그대로."""
    root = pretrained_root()
    if root is not None:
        rel = candidate[len(_PRETRAINED_PREFIX):] if candidate.startswith(_PRETRAINED_PREFIX) else candidate
        return root / rel
    return pathlib.Path(candidate)


def project_root_from(project_root: Optional[PathLike] = None) -> pathlib.Path:
    """저장소/작업 루트: 인자 → RSCTK_PROJECT_ROOT → cwd(표지 파일이 있으면) → 패키지 상위(개발 checkout) → cwd."""
    if project_root:
        return pathlib.Path(project_root).expanduser().resolve()
    env = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if env:
        return pathlib.Path(env).expanduser().resolve()
    cwd = pathlib.Path.cwd().resolve()
    if any((cwd / m).exists() for m in _ROOT_MARKERS):
        return cwd
    dev = pathlib.Path(__file__).resolve().parents[1]
    if (dev / "pyproject.toml").exists():
        return dev
    return cwd


__all__ = ["PRETRAINED_ROOT_ENV", "PROJECT_ROOT_ENV", "pretrained_root", "resolve_pretrained", "download_target", "project_root_from"]
