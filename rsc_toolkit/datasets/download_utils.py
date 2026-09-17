import hashlib
import json
import logging
import os
import pathlib
import time
import zipfile
from datetime import datetime, timezone
from typing import Optional, Union

import requests
import tqdm
import urllib3


logger = logging.getLogger(__name__)


# Give up on connecting quickly, but wait generously while data is flowing.
# (the ICBHI distribution server responds slowly and a 30 s read timeout often drops the connection)
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5
DEFAULT_CHUNK_SIZE = 1024 * 1024

# A file being downloaded is written to a temporary path with this suffix, not to the final path,
# so that "the file exists" implies "the download is complete".
PART_SUFFIX = ".part"

PathLike = Union[str, os.PathLike]


class DownloadError(RuntimeError):
    """Raised when a download cannot be completed even after retries."""


def _open_stream(url: str, *, headers: dict, connect_timeout: float, read_timeout: float):
    """Try with SSL verification first; retry without verification if the certificate is broken."""
    try:
        return requests.get(
            url,
            stream=True,
            headers=headers,
            timeout=(connect_timeout, read_timeout),
        )
    except requests.exceptions.SSLError:
        logger.warning("⚠️ SSL error. Retrying without certificate verification: %s", url)
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
    Download a file. Resume after interruptions and move to the final path only once complete.

    Args:
        expected_size: exact byte count, if known. Checked after completion.
        max_retries: number of retries on connection/read errors.
        resume: resume only the remaining part if the server supports Range.

    Returns:
        Path of the completed file.

    Raises:
        DownloadError: when all retries are exhausted or the size does not match the expected value.
    """
    logger.info("Starting download: %s -> %s", url, out_path)   # logged before the first attempt so a blocked network does not stay silent
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + PART_SUFFIX)

    if out.exists() and expected_size is not None and out.stat().st_size != expected_size:
        logger.warning(
            "Existing file size does not match expected (%d != %d). Re-downloading: %s",
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
                    # Range was requested although the file is already complete.
                    logger.info("Server returned 416. Treating the received content as complete: %s", out.name)
                    break

                r.raise_for_status()

                remaining = int(r.headers.get("content-length", 0)) or None

                if downloaded and r.status_code != 206:
                    # The server ignored Range and resent the whole file.
                    logger.warning("Server does not support resuming. Downloading from the start: %s", out.name)
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

            # If the stream reached its end, treat the download as complete.
            break

        except (requests.exceptions.RequestException, OSError) as exc:
            last_error = exc
            got = part.stat().st_size if part.exists() else 0
            if attempt >= max_retries:
                raise DownloadError(
                    f"Download failed after {max_retries} attempts: {url} "
                    f"(currently {got} bytes, partial file {part})"
                ) from exc
            backoff = min(2 ** (attempt - 1), 30)
            logger.warning(
                "Download interrupted (%d/%d): %s -- resuming in %.0f s from byte %d",
                attempt,
                max_retries,
                exc,
                backoff,
                got,
            )
            time.sleep(backoff)

    if not part.exists():
        raise DownloadError(f"Downloaded file not found: {part} (cause: {last_error})")

    size = part.stat().st_size

    # An expected size given by the caller takes precedence; otherwise compare with the size reported by the server.
    # Accept without verification only when neither is available.
    target = expected_size if expected_size is not None else advertised_total
    if target is not None and size != target:
        raise DownloadError(
            f"Download size does not match expected: {size} != {target} ({url}). "
            f"The partial file is kept at {part}; rerunning resumes the download."
        )

    # Only now does the file take the final path.
    os.replace(part, out)
    logger.info("✅ Download complete: %s (%d bytes)", out.name, size)
    return out


def sha256_of(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_download_provenance(
    path: PathLike,
    *,
    url: str,
    known_size: Optional[int] = None,
    known_sha256: Optional[str] = None,
    strict: bool = False,
) -> dict:
    """
    Record the size and sha256 of the downloaded file in <filename>.provenance.json and
    compare them with known values (known_*) if given.

    A mismatch with the known values means the upstream release has changed. By default this only warns and continues --
    the goal is to record "which data the experiments used", not to block the download.
    When changed data must never be used, pass strict=True to get a DownloadError.
    """
    p = pathlib.Path(path)
    size = p.stat().st_size
    digest = sha256_of(p)

    matches: Optional[bool] = None
    if known_size is not None or known_sha256 is not None:
        matches = (known_size is None or size == known_size) and (
            known_sha256 is None or digest == known_sha256
        )

    record = {
        "url": url,
        "file": p.name,
        "size": size,
        "sha256": digest,
        "known_size": known_size,
        "known_sha256": known_sha256,
        "matches_known": matches,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = p.with_name(p.name + ".provenance.json")
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    if matches is False:
        msg = (
            f"Upstream release differs from the known one: {p.name} size={size} sha256={digest[:12]}… "
            f"(known size={known_size}, sha256={(known_sha256 or '')[:12]}…). "
            f"The data may have changed; take care when comparing reproduction results. Record: {out}"
        )
        if strict:
            raise DownloadError(msg)
        logger.warning("⚠️ %s", msg)
    else:
        logger.info("Recorded download provenance: %s", out)
    return record


def is_valid_zip(zip_path: PathLike, *, deep: bool = False) -> bool:
    """
    Check that a zip file is complete.

    The zip central directory sits at the very end of the file. A partially downloaded zip lacks it
    and fails as soon as it is opened. No expected size is needed, which makes this a good completeness
    check, and only headers are read, so even a 2 GB file finishes instantly.

    With deep=True, the CRC of every entry is also checked (slow).
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
        logger.warning("Incomplete zip: %s (%s)", path, exc)
        return False


def extract_zip(zip_path: PathLike, out_dir: PathLike):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
