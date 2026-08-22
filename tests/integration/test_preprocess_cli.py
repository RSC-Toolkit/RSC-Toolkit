"""
`rsctk datasets preprocess` 통합 테스트.

이 명령의 존재 이유는 "학습 전에 피처 캐시를 미리 채우는 것" 이므로,
핵심 검증은 CLI 가 채운 캐시를 학습 경로(IndexDataModule)가 같은
디렉토리·키·sample_id 로 다시 읽는지(캐시 미스 0)다.

이전 구현은 두 겹으로 죽어 있었다:
- CLI 가 feature_processor 를 구성할 방법이 없어 항상 RuntimeError
- 호출하던 preprocess_features_from_index 는 주석 처리 상태
"""
import argparse
import pathlib

import pandas as pd
import pytest

import rsc_toolkit.datasets as datasets_cli
from rsc_toolkit.datasets.toy_dataset import ToyDataset


def _repo_root() -> pathlib.Path:
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("리포 루트를 찾지 못했다 (pyproject.toml 없음)")


REPO_ROOT = _repo_root()


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        command="datasets",
        dataset_command="preprocess",
        dataset="toy",
        data_config="toy_fbank",  # fbank: 외부 모델 다운로드 없이 빠르게 계산됨
        project_root=str(REPO_ROOT),
        data_dir=None,
        sample_level=None,
        workers=None,
        batch_size=None,
        cache_dir=None,
        force=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _load_split_map() -> dict:
    split_path = REPO_ROOT / "sample_data" / "toy_dataset" / "official_split.txt"
    mapping = {}
    for line in split_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1].lower()
    return mapping


@pytest.fixture
def toy_data_dir(tmp_path) -> pathlib.Path:
    """
    번들 toy 원본으로 tmp 에 인덱스를 만들고, 속도를 위해
    train 2개 + test 1개 recording 의 cycle 만 남긴다.
    """
    data_dir = tmp_path / "toy"
    proc = ToyDataset.make_processor(outputs_directory=str(data_dir), force=False)
    proc.build_index(sample_level="cycle")

    index_csv = data_dir / "index" / "index_cycle.csv"
    df = pd.read_csv(index_csv)

    split_map = _load_split_map()
    stems = df["recording_path"].map(lambda p: pathlib.Path(str(p)).stem)
    train_recs = sorted({s for s in stems if split_map.get(s) == "train"})[:2]
    test_recs = sorted({s for s in stems if split_map.get(s) == "test"})[:1]
    keep = set(train_recs) | set(test_recs)
    assert train_recs and test_recs, "픽스처 전제: 양쪽 split 에 recording 이 있어야 한다"

    df[stems.isin(keep)].to_csv(index_csv, index=False)
    return data_dir


def _training_datamodule(data_dir: pathlib.Path, cache_dir: pathlib.Path):
    """CLI 를 거치지 않고, 학습이 하는 그대로 datamodule 을 만든다."""
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule

    cfg = datasets_cli._compose_data_config("toy_fbank", REPO_ROOT)
    cfg["params"]["data_dir"] = str(data_dir)
    cfg["params"]["features"]["cache_dir"] = str(cache_dir)
    cfg["params"]["feature_precompute"] = False
    dm = build_datamodule(cfg)
    dm.setup("fit")
    return dm


def _assert_all_cached(dm) -> int:
    from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

    total = 0
    for name, ds in (("train", dm.train_ds), ("val", dm.val_ds)):
        sample_ids = IndexDataModule._collect_sample_ids(ds.index_ds)
        assert sample_ids, f"{name} split 이 비어 있다"
        missing = [s for s in sample_ids if not ds.fp.feature_cache.is_cached(s)]
        assert missing == [], f"{name} split 캐시 미스: {missing[:5]}..."
        total += len(sample_ids)
    return total


def test_preprocess_fills_cache_that_training_reuses(toy_data_dir, tmp_path):
    """
    핵심 검증: CLI 로 채운 캐시를 학습 경로가 캐시 미스 없이 읽는다.

    캐시 디렉토리나 키 규약이 조금이라도 어긋나면 여기서 미스가 난다.
    """
    cache_dir = tmp_path / "fc"

    datasets_cli.handle_command(
        _make_args(data_dir=str(toy_data_dir), cache_dir=str(cache_dir))
    )

    dm = _training_datamodule(toy_data_dir, cache_dir)
    total = _assert_all_cached(dm)

    stats = dm.train_ds.fp.feature_cache.get_cache_stats()
    assert stats["pt_files"] == total  # train+val(=test) 전체가 한 키 디렉토리에


def test_preprocess_force_recomputes(toy_data_dir, tmp_path):
    """--force 는 같은 설정의 기존 캐시를 비우고 전부 다시 채운다."""
    cache_dir = tmp_path / "fc"
    datasets_cli.handle_command(
        _make_args(data_dir=str(toy_data_dir), cache_dir=str(cache_dir))
    )

    # 캐시 파일 하나를 손상시켜 두면, force 재계산이 실제로 일어났는지 보인다
    features_dirs = list((cache_dir / "features").iterdir())
    assert len(features_dirs) == 1
    pt_files = sorted(features_dirs[0].glob("*.pt"))
    corrupted = pt_files[0]
    corrupted.write_bytes(b"broken")

    datasets_cli.handle_command(
        _make_args(data_dir=str(toy_data_dir), cache_dir=str(cache_dir), force=True)
    )

    # 손상 파일이 새로 계산돼 정상 텐서로 복원됐어야 한다
    import torch

    tensor = torch.load(corrupted, map_location="cpu")
    assert torch.is_tensor(tensor)

    dm = _training_datamodule(toy_data_dir, cache_dir)
    _assert_all_cached(dm)


def test_preprocess_rejects_mismatched_data_config(toy_data_dir, tmp_path):
    """positional dataset 과 config 의 dataset_name 이 다르면 거부한다."""
    with pytest.raises(ValueError, match="dataset_name"):
        datasets_cli.handle_command(
            _make_args(
                dataset="icbhi",
                data_config="toy_fbank",
                data_dir=str(toy_data_dir),
                cache_dir=str(tmp_path / "fc"),
            )
        )


def test_preprocess_unknown_data_config_lists_available(tmp_path):
    with pytest.raises(ValueError, match="toy_fbank"):
        datasets_cli.handle_command(
            _make_args(data_config="does_not_exist", data_dir=str(tmp_path))
        )


def test_preprocess_missing_index_points_to_build_index(tmp_path):
    """인덱스가 없으면 build-index 를 안내하며 실패한다."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="build-index"):
        datasets_cli.handle_command(
            _make_args(data_dir=str(empty), cache_dir=str(tmp_path / "fc"))
        )


def test_preprocess_rejects_config_without_features(toy_data_dir, tmp_path, monkeypatch):
    """features 가 꺼진 config 로는 뽑을 것이 없으므로 명확히 거부한다."""
    real_compose = datasets_cli._compose_data_config

    def fake_compose(name, project_root):
        cfg = real_compose(name, project_root)
        cfg["params"]["features"]["enabled"] = False
        return cfg

    monkeypatch.setattr(datasets_cli, "_compose_data_config", fake_compose)

    with pytest.raises(ValueError, match="features"):
        datasets_cli.handle_command(
            _make_args(data_dir=str(toy_data_dir), cache_dir=str(tmp_path / "fc"))
        )
