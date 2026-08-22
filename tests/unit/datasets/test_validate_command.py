"""
`rsctk datasets validate` 의 판정/종료 코드 동작 테스트.

이전에는 검증 함수의 반환값을 버리고 무조건 "✅ passed" 를 찍어서,
wav 가 0 개인 디렉토리도 통과로 보고했다.
"""
import argparse

import pytest

import rsc_toolkit.datasets as datasets_cli


class _Downloader:
    def __init__(self, ok):
        self._ok = ok

    def validate_downloaded_files(self):
        return self._ok


class _Processor:
    """raw_data_directory 를 가져야 CLI 가 경로를 주입한다."""

    def __init__(self, result):
        self._result = result
        self.raw_data_directory = None

    def validate_processed_files(self):
        return self._result


class _Dataset:
    def __init__(self, downloaded_ok, processed_result):
        self.downloader = _Downloader(downloaded_ok)
        self.processor = _Processor(processed_result)


def _run_validate(monkeypatch, tmp_path, *, downloaded_ok, processed_result):
    monkeypatch.setattr(
        datasets_cli,
        "create_dataset",
        lambda args: _Dataset(downloaded_ok, processed_result),
    )
    args = argparse.Namespace(
        command="datasets",
        dataset_command="validate",
        dataset="icbhi",
        data_dir=str(tmp_path),
        project_root=None,
        force=False,
    )
    return datasets_cli.handle_command(args)


def test_validate_fails_when_download_invalid(monkeypatch, tmp_path):
    """다운로드 검증이 False 면 실패(0 이 아닌 종료 코드)여야 한다."""
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=False, processed_result=None
    )
    assert rc != 0


def test_validate_passes_when_download_valid(monkeypatch, tmp_path):
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=True, processed_result=True
    )
    assert rc == 0


def test_unimplemented_processed_check_is_not_a_failure(monkeypatch, tmp_path):
    """
    validate_processed_files 가 None(미구현) 이면 실패로 치지 않는다.

    현재 icbhi/sprsound/hflungv1 processor 는 모두 미구현 스텁이므로,
    이걸 실패로 처리하면 정상 데이터셋도 전부 실패로 뜬다.
    """
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=True, processed_result=None
    )
    assert rc == 0


def test_validate_fails_when_processed_invalid(monkeypatch, tmp_path):
    """미구현(None) 과 달리 명시적 False 는 실패로 처리한다."""
    rc = _run_validate(
        monkeypatch, tmp_path, downloaded_ok=True, processed_result=False
    )
    assert rc != 0


@pytest.mark.parametrize("dataset_name", ["icbhi", "sprsound", "hflungv1"])
def test_processor_stubs_return_none_not_false(dataset_name):
    """
    미구현 스텁이 False 가 아니라 None 을 돌려주는지 확인한다.

    False 를 돌려주면 위 로직에서 '검증 실패' 로 잘못 보고된다.
    """
    from rsc_toolkit.datasets.factory import DatasetFactory

    cls = DatasetFactory.get_dataset_class(dataset_name=dataset_name)
    processor = cls.make_processor(outputs_directory="/tmp/rsctk-validate-test", force=False)
    assert processor.validate_processed_files() is None
