import logging
import os
import pathlib
import time
import zipfile
from typing import Optional, Union

import requests
import tqdm
import urllib3


logger = logging.getLogger(__name__)


# 연결은 빨리 포기하되, 데이터가 흐르는 중에는 넉넉히 기다린다.
# (ICBHI 배포 서버는 응답이 느려 read timeout 30초로는 자주 끊긴다)
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5
DEFAULT_CHUNK_SIZE = 1024 * 1024

# 받는 중인 파일은 최종 경로가 아니라 이 접미사를 붙인 임시 경로에 쓴다.
# 그래야 "파일이 있다 = 다 받았다" 가 성립한다.
PART_SUFFIX = ".part"

PathLike = Union[str, os.PathLike]


class DownloadError(RuntimeError):
    """다운로드가 재시도 후에도 끝내 완료되지 못했을 때."""


def _open_stream(url: str, *, headers: dict, connect_timeout: float, read_timeout: float):
    """SSL 검증을 먼저 시도하고, 인증서가 깨져 있으면 검증 없이 재시도한다."""
    try:
        return requests.get(
            url,
            stream=True,
            headers=headers,
            timeout=(connect_timeout, read_timeout),
        )
    except requests.exceptions.SSLError:
        logger.warning("⚠️ SSL 오류. 인증서 검증 없이 재시도한다: %s", url)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return requests.get(
            url,
            stream=True,
            headers=headers,
            verify=False,
            timeout=(connect_timeout, read_timeout),
        )


def download_file(
    url: str,
    out_path: PathLike,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    expected_size: Optional[int] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    resume: bool = True,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> pathlib.Path:
    """
    파일을 내려받는다. 중단되면 이어받고, 다 받은 뒤에만 최종 경로로 옮긴다.

    Args:
        expected_size: 알고 있는 정확한 바이트 수. 주면 완료 후 대조한다.
        max_retries: 연결/읽기 오류 시 재시도 횟수.
        resume: 서버가 Range 를 지원하면 남은 부분만 이어받는다.

    Returns:
        완성된 파일 경로.

    Raises:
        DownloadError: 재시도를 모두 소진했거나 크기가 기대치와 다를 때.
    """
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + PART_SUFFIX)

    if out.exists() and expected_size is not None and out.stat().st_size != expected_size:
        logger.warning(
            "기존 파일 크기가 기대치와 다르다 (%d != %d). 다시 받는다: %s",
            out.stat().st_size,
            expected_size,
            out,
        )
        out.unlink()

    last_error: Optional[BaseException] = None
    advertised_total: Optional[int] = None

    for attempt in range(1, max_retries + 1):
        downloaded = part.stat().st_size if part.exists() else 0

        headers = {}
        if resume and downloaded:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            with _open_stream(
                url,
                headers=headers,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            ) as r:
                if r.status_code == 416:
                    # 이미 다 받은 상태에서 Range 를 요청한 경우.
                    logger.info("서버가 416 을 반환했다. 받은 내용을 완료로 본다: %s", out.name)
                    break

                r.raise_for_status()

                remaining = int(r.headers.get("content-length", 0)) or None

                if downloaded and r.status_code != 206:
                    # 서버가 Range 를 무시하고 전체를 다시 보냈다.
                    logger.warning("서버가 이어받기를 지원하지 않는다. 처음부터 받는다: %s", out.name)
                    downloaded = 0

                total = (remaining + downloaded) if remaining is not None else None
                if total is not None:
                    advertised_total = total
                mode = "ab" if downloaded else "wb"

                with open(part, mode) as f, tqdm.tqdm(
                    total=total,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    desc=out.name,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            # 스트림이 끝까지 왔으면 완료로 본다.
            break

        except (requests.exceptions.RequestException, OSError) as exc:
            last_error = exc
            got = part.stat().st_size if part.exists() else 0
            if attempt >= max_retries:
                raise DownloadError(
                    f"{max_retries}회 시도 후에도 다운로드 실패: {url} "
                    f"(현재 {got} bytes, 이어받기 파일 {part})"
                ) from exc
            backoff = min(2 ** (attempt - 1), 30)
            logger.warning(
                "다운로드 중단 (%d/%d회): %s — %.0f초 후 %d bytes 지점부터 이어받는다",
                attempt,
                max_retries,
                exc,
                backoff,
                got,
            )
            time.sleep(backoff)

    if not part.exists():
        raise DownloadError(f"다운로드 결과 파일이 없다: {part} (원인: {last_error})")

    size = part.stat().st_size

    # 호출자가 준 기대 크기가 우선이고, 없으면 서버가 알려준 크기와 대조한다.
    # 둘 다 없을 때만 검증 없이 받아들인다.
    target = expected_size if expected_size is not None else advertised_total
    if target is not None and size != target:
        raise DownloadError(
            f"다운로드 크기가 기대치와 다르다: {size} != {target} ({url}). "
            f"받다 만 파일을 {part} 에 남겨두니 다시 실행하면 이어받는다."
        )

    # 여기까지 왔을 때만 최종 경로를 차지한다.
    os.replace(part, out)
    logger.info("✅ 다운로드 완료: %s (%d bytes)", out.name, size)
    return out


def is_valid_zip(zip_path: PathLike, *, deep: bool = False) -> bool:
    """
    zip 이 끝까지 온전한지 본다.

    zip 의 중앙 디렉토리는 파일 맨 뒤에 있다. 받다 만 zip 은 이게 없어서
    여는 순간 걸린다. 기대 크기를 몰라도 되므로 완료 판정에 쓰기 좋고,
    헤더만 읽으므로 2GB 짜리도 즉시 끝난다.

    deep=True 면 전 항목의 CRC 까지 대조한다(느리다).
    """
    path = pathlib.Path(zip_path)
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, "r") as z:
            if not z.namelist():
                return False
            if deep:
                return z.testzip() is None
            return True
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("온전하지 않은 zip: %s (%s)", path, exc)
        return False


def extract_zip(zip_path: PathLike, out_dir: PathLike):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
