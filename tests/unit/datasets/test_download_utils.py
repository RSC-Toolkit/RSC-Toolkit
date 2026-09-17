"""
Tests for resume/retry/integrity behavior of download_file.

The previous implementation wrote directly to the final path, with a 30 s read timeout and no retries.
So an interrupted download 'left a partial file at the final path', and the next run
saw only that the file existed, took it as complete and tried to extract a broken archive.
"""
import http.server
import threading
import zipfile
import io
import pathlib

import pytest

from rsc_toolkit.datasets.download_utils import (
    DownloadError,
    PART_SUFFIX,
    download_file,
    is_valid_zip,
    record_download_provenance,
    sha256_of,
)


PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


class _Handler(http.server.BaseHTTPRequestHandler):
    """Test server that supports Range and drops the connection mid-transfer on demand."""

    # state manipulated by the tests
    payload = PAYLOAD
    accept_ranges = True
    truncate_after = None  # send only this many bytes, then drop the connection
    fail_times = 0  # remaining number of 'drops'
    request_log = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        cls = type(self)
        cls.request_log.append(self.headers.get("Range"))

        start = 0
        rng = self.headers.get("Range")
        if rng and cls.accept_ranges:
            start = int(rng.split("=")[1].split("-")[0])

        body = cls.payload[start:]

        if start and cls.accept_ranges:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(cls.payload)-1}/{len(cls.payload)}")
        else:
            self.send_response(200)
            body = cls.payload
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes" if cls.accept_ranges else "none")
        self.end_headers()

        if cls.fail_times > 0:
            cls.fail_times -= 1
            cut = cls.truncate_after or (len(body) // 2)
            self.wfile.write(body[:cut])
            self.wfile.flush()
            # close the connection without sending the rest to cause a read error on the client.
            self.close_connection = True
            return

        self.wfile.write(body)


@pytest.fixture
def server():
    _Handler.payload = PAYLOAD
    _Handler.accept_ranges = True
    _Handler.truncate_after = None
    _Handler.fail_times = 0
    _Handler.request_log = []

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/file.bin"
    httpd.shutdown()
    httpd.server_close()


def test_downloads_complete_file(server, tmp_path):
    out = tmp_path / "file.bin"
    download_file(server, out, max_retries=2)

    assert out.read_bytes() == PAYLOAD
    assert not out.with_name(out.name + PART_SUFFIX).exists()


def test_partial_download_never_occupies_final_path(server, tmp_path):
    """
    When interrupted, nothing may remain at the final path.

    This is what makes the next run's 'file exists means complete' check valid.
    """
    _Handler.fail_times = 99  # keep dropping
    out = tmp_path / "file.bin"

    with pytest.raises(DownloadError):
        download_file(server, out, chunk_size=8192, max_retries=2)

    assert not out.exists()
    # the partial file stays in .part so it can be resumed
    assert out.with_name(out.name + PART_SUFFIX).exists()


def test_resumes_after_interruption(server, tmp_path):
    """Even after one drop, the retry resumes with Range and completes."""
    _Handler.fail_times = 1
    out = tmp_path / "file.bin"

    download_file(server, out, chunk_size=8192, max_retries=3)

    assert out.read_bytes() == PAYLOAD
    # the second request must carry a Range header
    assert _Handler.request_log[0] is None
    assert _Handler.request_log[1] is not None
    assert _Handler.request_log[1].startswith("bytes=")


def test_restarts_when_server_ignores_range(server, tmp_path):
    """If the server ignores Range, download again from the start and produce an intact file."""
    _Handler.fail_times = 1
    _Handler.accept_ranges = False
    out = tmp_path / "file.bin"

    download_file(server, out, chunk_size=8192, max_retries=3)

    assert out.read_bytes() == PAYLOAD


def test_size_mismatch_is_rejected(server, tmp_path):
    """If the size differs from the expected one, the file is not moved to the final path."""
    out = tmp_path / "file.bin"

    with pytest.raises(DownloadError, match="size does not match expected"):
        download_file(server, out, expected_size=len(PAYLOAD) + 1, max_retries=1)

    assert not out.exists()


def test_stale_file_with_wrong_size_is_redownloaded(server, tmp_path):
    """If the expected size is known and the existing file differs, download again."""
    out = tmp_path / "file.bin"
    out.write_bytes(b"truncated")

    download_file(server, out, expected_size=len(PAYLOAD), max_retries=2)

    assert out.read_bytes() == PAYLOAD


def _make_zip(path: pathlib.Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "hello")
        z.writestr("b.txt", "world")
    path.write_bytes(buf.getvalue())


def test_is_valid_zip_accepts_complete_zip(tmp_path):
    p = tmp_path / "ok.zip"
    _make_zip(p)
    assert is_valid_zip(p) is True


def test_is_valid_zip_rejects_truncated_zip(tmp_path):
    """
    A partially downloaded zip has no central directory and must be filtered out.

    Without this check, 'file exists = complete' leads to extracting a broken archive.
    """
    p = tmp_path / "ok.zip"
    _make_zip(p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])

    assert is_valid_zip(p) is False


def test_is_valid_zip_rejects_missing_file(tmp_path):
    assert is_valid_zip(tmp_path / "nope.zip") is False


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "blob.bin"
    p.write_bytes(PAYLOAD)
    assert sha256_of(p) == hashlib.sha256(PAYLOAD).hexdigest()


def test_provenance_is_recorded_and_matches_known_values(tmp_path):
    import hashlib
    import json

    p = tmp_path / "data.zip"
    p.write_bytes(PAYLOAD)
    rec = record_download_provenance(
        p, url="http://x/data.zip",
        known_size=len(PAYLOAD), known_sha256=hashlib.sha256(PAYLOAD).hexdigest(),
    )
    assert rec["matches_known"] is True
    saved = json.loads((tmp_path / "data.zip.provenance.json").read_text())
    assert saved["size"] == len(PAYLOAD) and saved["sha256"] == rec["sha256"]
    assert saved["url"] == "http://x/data.zip"


def test_provenance_warns_but_continues_on_upstream_change(tmp_path, caplog):
    """
    A size/hash differing from the known values means upstream changed. The default policy is warn + record, not reject —
    the purpose is to record which data the experiments used.
    """
    import logging

    p = tmp_path / "data.zip"
    p.write_bytes(PAYLOAD)
    with caplog.at_level(logging.WARNING):
        rec = record_download_provenance(p, url="u", known_size=len(PAYLOAD) + 1)
    assert rec["matches_known"] is False
    assert any("Upstream release differs" in r.message for r in caplog.records)
    assert (tmp_path / "data.zip.provenance.json").exists()


def test_provenance_strict_mode_raises_on_upstream_change(tmp_path):
    p = tmp_path / "data.zip"
    p.write_bytes(PAYLOAD)
    with pytest.raises(DownloadError, match="Upstream release differs"):
        record_download_provenance(p, url="u", known_size=len(PAYLOAD) + 1, strict=True)


def test_provenance_without_known_values_just_records(tmp_path):
    p = tmp_path / "snap.zip"
    p.write_bytes(PAYLOAD)
    rec = record_download_provenance(p, url="u")
    assert rec["matches_known"] is None
