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
    feature_dim: int = 128
    
    def _get_processor_config(self) -> Dict[str, Any]:
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


@pytest.fixture
def toy_dataset_path():
    """Get path to toy dataset"""
    project_root = pathlib.Path(__file__).parent.parent.parent
    toy_path = project_root / "sample_data" / "toy_dataset" / "files"
    if not toy_path.exists():
        pytest.skip(f"Toy dataset not found at {toy_path}")
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
            cache_features=True,
            cache_dir=cache_dir
        ),
        num_preprocess_workers=2,  # Use 2 workers for testing
        force=True  # Force reprocessing
    )
    
    # Create dataset
    dataset = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor
    )
    
    # Process dataset (this will preprocess features)
    logger.info("Starting dataset processing with feature preprocessing...")
    dataset.process()
    
    # Verify cache was created
    assert processor.feature_processor._feature_cache is not None
    
    # Check cache stats
    cache_stats = processor.feature_processor._feature_cache.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")
    
    # Should have cached some samples
    assert cache_stats["total_entries"] > 0
    assert cache_stats["existing_files"] > 0
    
    # Verify cache directory structure
    cache_features_dir = pathlib.Path(cache_stats["cache_dir"])
    assert cache_features_dir.exists()
    assert (cache_features_dir / "cache_metadata.json").exists()


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
            cache_features=True,
            cache_dir=cache_dir
        ),
        num_preprocess_workers=2,
        force=True
    )
    
    dataset1 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor1
    )
    
    dataset1.process()
    first_run_stats = processor1.feature_processor._feature_cache.get_cache_stats()
    first_run_count = first_run_stats["total_entries"]
    
    # Second run - should reuse cache
    processor2 = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=128,
            cache_features=True,
            cache_dir=cache_dir
        ),
        num_preprocess_workers=2,
        force=False  # Don't force, should use cache
    )
    
    dataset2 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor2
    )
    
    dataset2.process()
    second_run_stats = processor2.feature_processor._feature_cache.get_cache_stats()
    
    # Should have same number of cached entries
    assert second_run_stats["total_entries"] == first_run_count
    assert second_run_stats["existing_files"] == first_run_count


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
            cache_features=True,
            cache_dir=cache_dir
        ),
        num_preprocess_workers=2,
        force=True
    )
    
    dataset1 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor1
    )
    dataset1.process()
    
    cache1_key = processor1.feature_processor._feature_cache.cache_key
    
    # Processor with feature_dim=256 (different config)
    processor2 = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=True,
        feature_processor=SimpleTestProcessor(
            feature_dim=256,  # Different dimension
            cache_features=True,
            cache_dir=cache_dir
        ),
        num_preprocess_workers=2,
        force=True
    )
    
    dataset2 = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor2
    )
    dataset2.process()
    
    cache2_key = processor2.feature_processor._feature_cache.cache_key
    
    # Should have different cache keys (different processor configs)
    assert cache1_key != cache2_key
    
    # Should have different cache directories
    cache1_dir = processor1.feature_processor._feature_cache.features_dir
    cache2_dir = processor2.feature_processor._feature_cache.features_dir
    assert cache1_dir != cache2_dir
    
    # Both should have cached samples
    stats1 = processor1.feature_processor._feature_cache.get_cache_stats()
    stats2 = processor2.feature_processor._feature_cache.get_cache_stats()
    assert stats1["total_entries"] > 0
    assert stats2["total_entries"] > 0

