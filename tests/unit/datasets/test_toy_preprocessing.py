"""
Slow test for toy dataset preprocessing with feature caching.
This test actually processes audio files and caches features.

Run with: pytest tests/datasets/test_toy_preprocessing.py -m slow
Or skip with: pytest -m "not slow"
"""
import pytest
import pathlib
import torch
import logging
from rsc_toolkit.datasets.toy_dataset import ToyDataset
from rsc_toolkit.processors.base import BaseAudioProcessor
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class SimpleTestProcessor(BaseAudioProcessor):
    """Simple processor for testing - computes mean features"""

    def __init__(self, feature_dim: int = 128, **kwargs):
        # BaseAudioProcessor 는 pydantic 모델이 아니라 평범한 ABC 다.
        # 클래스 어노테이션만 적어두면 생성자 인자로 받아지지 않는다.
        super().__init__(**kwargs)
        self.feature_dim = feature_dim

    def get_processor_config(self) -> Dict[str, Any]:
        return {"feature_dim": self.feature_dim}

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        """
        Simple feature extraction: mean pooling with dimension expansion.
        This is fast and doesn't require external dependencies.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute mean across time dimension
        features = audio.mean(dim=-1, keepdim=True)
        
        # Expand/repeat to desired dimension
        if features.shape[-1] < self.feature_dim:
            # Repeat to get at least feature_dim elements
            repeat_factor = (self.feature_dim // features.shape[-1]) + 1
            features = features.repeat(1, repeat_factor)
        
        # Take first feature_dim elements
        return features[:, :self.feature_dim]


def _repo_root() -> pathlib.Path:
    """pyproject.toml 이 있는 디렉토리를 리포 루트로 본다."""
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("리포 루트를 찾지 못했다 (pyproject.toml 없음)")


@pytest.fixture
def toy_dataset_path():
    """
    번들된 toy 데이터셋 경로.

    이전에는 parent 를 한 번 덜 올라가 tests/sample_data/... 를 찾았고,
    없으니 pytest.skip 으로 빠져 3건이 조용히 넘어갔다. 데이터는 리포에
    항상 들어 있으므로 없으면 skip 이 아니라 실패로 다뤄야 한다.
    """
    toy_path = _repo_root() / "sample_data" / "toy_dataset" / "files"
    assert toy_path.exists(), f"번들된 toy 데이터셋이 없다: {toy_path}"
    return toy_path


@pytest.mark.slow
def test_toy_dataset_preprocessing(tmp_path, toy_dataset_path):
    """Test preprocessing toy dataset with feature caching"""
    cache_dir = str(tmp_path / "cache")
    
    # Create processor with preprocessing enabled
    processor = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=128,
        ),
        num_preprocess_workers=2,  # Use 2 workers for testing
        force=True  # Force reprocessing
    )
    
    # Create dataset
    dataset = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor
    )
    
    # process_dataset() 은 인덱스만 만들고 피처 전처리는 하지 않는다.
    # 전처리는 preprocess_all_features() 가 담당하므로 직접 부른다.
    dataset.process()
    processor.preprocess_all_features(processor._discover_recordings())
    
    # Verify cache was created
    assert processor.feature_processor.feature_cache is not None
    
    # Check cache stats
    cache_stats = processor.feature_processor.feature_cache.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")
    
    # Should have cached some samples
    assert cache_stats["pt_files"] > 0
    
    # Verify cache directory structure
    cache_features_dir = pathlib.Path(cache_stats["cache_dir"])
    assert cache_features_dir.exists()
    # FeatureCache 는 metadata 파일을 두지 않고 디렉토리를 스캔한다.
    assert list(cache_features_dir.glob("*.pt"))


@pytest.mark.slow
def test_toy_dataset_preprocessing_cache_reuse(tmp_path, toy_dataset_path):
    """Test that cached features are reused on second run"""
    cache_dir = str(tmp_path / "cache")
    
    # First run - process and cache
    processor1 = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=128,
        ),
        num_preprocess_workers=2,
        force=True
    )
    
    dataset1 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor1
    )
    
    dataset1.process()
    processor1.preprocess_all_features(processor1._discover_recordings())
    first_run_stats = processor1.feature_processor.feature_cache.get_cache_stats()
    first_run_count = first_run_stats["pt_files"]
    
    # Second run - should reuse cache
    processor2 = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=128,
        ),
        num_preprocess_workers=2,
        force=False  # Don't force, should use cache
    )
    
    dataset2 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor2
    )
    
    dataset2.process()
    processor2.preprocess_all_features(processor2._discover_recordings())
    second_run_stats = processor2.feature_processor.feature_cache.get_cache_stats()
    
    # Should have same number of cached entries
    assert second_run_stats["pt_files"] == first_run_count


@pytest.mark.slow
def test_toy_dataset_different_processors_isolated_cache(tmp_path, toy_dataset_path):
    """Test that different processors use isolated caches"""
    cache_dir = str(tmp_path / "cache")
    
    # Processor with feature_dim=128
    processor1 = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=128,
        ),
        num_preprocess_workers=2,
        force=True
    )
    
    dataset1 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor1
    )
    dataset1.process()
    processor1.preprocess_all_features(processor1._discover_recordings())

    cache1_key = processor1.feature_processor.feature_cache.cache_key
    
    # Processor with feature_dim=256 (different config)
    processor2 = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=256,  # Different dimension
        ),
        num_preprocess_workers=2,
        force=True
    )
    
    dataset2 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor2
    )
    dataset2.process()
    processor2.preprocess_all_features(processor2._discover_recordings())

    cache2_key = processor2.feature_processor.feature_cache.cache_key
    
    # Should have different cache keys (different processor configs)
    assert cache1_key != cache2_key
    
    # Should have different cache directories
    cache1_dir = processor1.feature_processor.feature_cache.features_dir
    cache2_dir = processor2.feature_processor.feature_cache.features_dir
    assert cache1_dir != cache2_dir
    
    # Both should have cached samples
    stats1 = processor1.feature_processor.feature_cache.get_cache_stats()
    stats2 = processor2.feature_processor.feature_cache.get_cache_stats()
    assert stats1["pt_files"] > 0
    assert stats2["pt_files"] > 0

