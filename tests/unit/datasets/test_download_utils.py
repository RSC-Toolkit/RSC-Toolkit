"""
download_file 의 이어받기/재시도/무결성 동작 테스트.

이전 구현은 최종 경로에 바로 쓰고, read timeout 30 초에 재시도도 없었다.
그래서 중간에 끊기면 '받다 만 파일이 최종 경로에 남고', 다음 실행이
파일 존재만 보고 완료로 오인해 깨진 압축 해제를 시도했다.
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
)


PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


class _Handler(http.server.BaseHTTPRequestHandler):
    """Range 를 지원하고, 지시에 따라 중간에 끊는 시험용 서버."""

    # 테스트가 조작하는 상태
    payload = PAYLOAD
    accept_ranges = True
    truncate_after = None  # 이 바이트 수만 보내고 연결을 끊는다
    fail_times = 0  # 남은 '끊을 횟수'
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
            # 나머지를 보내지 않고 연결을 닫아 클라이언트에서 읽기 오류를 낸다.
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
    중단되면 최종 경로에는 아무것도 남지 않아야 한다.

    이게 보장돼야 다음 실행의 '파일이 있으면 완료' 판정이 성립한다.
    """
    _Handler.fail_times = 99  # 계속 끊는다
    out = tmp_path / "file.bin"

    with pytest.raises(DownloadError):
        download_file(server, out, chunk_size=8192, max_retries=2)

    assert not out.exists()
    # 이어받을 수 있도록 부분 파일은 .part 에 남는다
    assert out.with_name(out.name + PART_SUFFIX).exists()


def test_resumes_after_interruption(server, tmp_path):
    """한 번 끊겨도 재시도에서 Range 로 이어받아 완성한다."""
    _Handler.fail_times = 1
    out = tmp_path / "file.bin"

    download_file(server, out, chunk_size=8192, max_retries=3)

    assert out.read_bytes() == PAYLOAD
    # 두 번째 요청은 Range 헤더를 달고 나가야 한다
    assert _Handler.request_log[0] is None
    assert _Handler.request_log[1] is not None
    assert _Handler.request_log[1].startswith("bytes=")


def test_restarts_when_server_ignores_range(server, tmp_path):
    """서버가 Range 를 무시하면 처음부터 다시 받아 온전한 파일을 만든다."""
    _Handler.fail_times = 1
    _Handler.accept_ranges = False
    out = tmp_path / "file.bin"

    download_file(server, out, chunk_size=8192, max_retries=3)

    assert out.read_bytes() == PAYLOAD


def test_size_mismatch_is_rejected(server, tmp_path):
    """기대 크기와 다르면 최종 경로로 옮기지 않는다."""
    out = tmp_path / "file.bin"

    with pytest.raises(DownloadError, match="크기가 기대치와 다르다"):
        download_file(server, out, expected_size=len(PAYLOAD) + 1, max_retries=1)

    assert not out.exists()


def test_stale_file_with_wrong_size_is_redownloaded(server, tmp_path):
    """기대 크기를 아는데 기존 파일이 그와 다르면 다시 받는다."""
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
    받다 만 zip 은 중앙 디렉토리가 없어 걸러져야 한다.

    이 판정이 없으면 '파일이 있다 = 완료' 로 보고 깨진 압축 해제를 시도한다.
    """
    p = tmp_path / "ok.zip"
    _make_zip(p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])

    assert is_valid_zip(p) is False


def test_is_valid_zip_rejects_missing_file(tmp_path):
    assert is_valid_zip(tmp_path / "nope.zip") is False
