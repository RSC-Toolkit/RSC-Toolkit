"""
`rsctk datasets validate` 의 판정/종료 코드 동작 테스트.

이전에는 검증 함수의 반환값을 버리고 무조건 "✅ passed" 를 찍어서,
wav 가 0 개인 디렉토리도 통과로 보고했다.
"""
import argparse
import pathlib

import pytest

from rsc_toolkit.datasets import cli as datasets_cli  # monkeypatch 대상은 CLI 모듈 (P1-8 분리)


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
def test_processors_report_none_when_nothing_processed(dataset_name, tmp_path):
    """
    아직 build-index 를 안 한 빈 출력 디렉토리에서는 None(검사할 것 없음)이어야 한다.

    False 를 돌려주면 위 로직에서 '검증 실패' 로 잘못 보고된다.
    """
    from rsc_toolkit.datasets.factory import DatasetFactory

    cls = DatasetFactory.get_dataset_class(dataset_name=dataset_name)
    processor = cls.make_processor(outputs_directory=str(tmp_path), force=False)
    assert processor.validate_processed_files() is None


# ---------------------------------------------------------------------------
# 인덱스 산출물 실제 검증 (base 기본 구현 + 데이터셋 훅)
# ---------------------------------------------------------------------------

def _repo_root():
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


@pytest.fixture
def icbhi_processor_with_index(tmp_path):
    """번들 toy 원본(ICBHI 형식)으로 실제 cycle 인덱스를 만든 ICBHI processor."""
    from rsc_toolkit.datasets.icbhi.processor import ICBHIDatasetProcessor

    proc = ICBHIDatasetProcessor(
        outputs_directory=str(tmp_path),
        raw_data_directory=str(_repo_root() / "sample_data" / "toy_dataset" / "files"),
    )
    proc.build_index(sample_level="cycle")
    return proc


def test_real_index_validates_true(icbhi_processor_with_index):
    assert icbhi_processor_with_index.validate_processed_files() is True


def test_index_with_zero_rows_is_invalid(icbhi_processor_with_index, tmp_path):
    """
    build-index 가 exit 0 으로 끝나도 행이 0 이면 학습이 불가능하다.
    (SPRSound 에서 실제로 겪었다 — 문서 §5 신규 B)
    """
    index_csv = tmp_path / "index" / "index_cycle.csv"
    header = index_csv.read_text().splitlines()[0]
    index_csv.write_text(header + "\n")
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_index_missing_required_column_is_invalid(icbhi_processor_with_index, tmp_path):
    import pandas as pd

    index_csv = tmp_path / "index" / "index_cycle.csv"
    df = pd.read_csv(index_csv).drop(columns=["label"])
    df.to_csv(index_csv, index=False)
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_stale_stats_file_is_invalid(icbhi_processor_with_index, tmp_path):
    """stats 의 total_samples 와 행수가 다르면 인덱스와 stats 가 따로 논다는 뜻이다."""
    import json

    stats = tmp_path / "index" / "index_cycle_stats.json"
    data = json.loads(stats.read_text())
    data["total_samples"] = 1
    stats.write_text(json.dumps(data))
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_only_existing_levels_are_checked(icbhi_processor_with_index):
    """cycle 만 만들고 clip 은 안 만들어도 실패가 아니다 (존재하는 레벨만 검사)."""
    assert icbhi_processor_with_index.validate_processed_files() is True


def test_cycle_index_with_missing_labels_is_invalid(icbhi_processor_with_index, tmp_path):
    import pandas as pd

    index_csv = tmp_path / "index" / "index_cycle.csv"
    df = pd.read_csv(index_csv)
    df.loc[0, "label"] = None
    df.to_csv(index_csv, index=False)
    assert icbhi_processor_with_index.validate_processed_files() is False


def test_sprsound_requires_official_split_next_to_index(tmp_path):
    """데이터셋별 보조 산출물 훅: sprsound 는 index 옆에 official_split.txt 가 있어야 한다."""
    from rsc_toolkit.datasets.sprsound.processor import SPRSoundDatasetProcessor

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_cycle.csv").write_text(
        "sample_id,recording_path,sample_level,t0,t1,split,label,meta\n"
        "a_0.0_1.0,a.wav,cycle,0.0,1.0,train,0,{}\n"
    )
    proc = SPRSoundDatasetProcessor(outputs_directory=str(tmp_path))

    assert proc.validate_processed_files() is False  # 보조 파일 없음

    (index_dir / "official_split.txt").write_text("a train\n")
    assert proc.validate_processed_files() is True


def test_feature_cache_validate_entries_is_shape_agnostic(tmp_path):
    """
    피처 캐시 검증은 모양을 가정하지 않는다 — 모델/피처 종류마다 다르기 때문이다.
    존재·로드 가능 여부만 본다.
    """
    import torch
    from rsc_toolkit.datasets.feature_cache import FeatureCache

    cache = FeatureCache(cache_dir=tmp_path)
    cache.configure(processor_config={"kind": "any"}, processor_name="AnyProcessor")
    cache.put("clap_like", torch.randn(1, 512))
    cache.put("fbank_like", torch.randn(1024, 128))
    cache.put("broken", torch.randn(3))
    cache._get_cache_path("broken").write_bytes(b"not a tensor")

    report = cache.validate_entries(["clap_like", "fbank_like", "broken", "never_cached"])
    assert report["total"] == 4
    assert report["missing"] == ["never_cached"]
    assert report["corrupted"] == ["broken"]
    assert report["ok"] is False

    assert cache.validate_entries(["clap_like", "fbank_like"])["ok"] is True
