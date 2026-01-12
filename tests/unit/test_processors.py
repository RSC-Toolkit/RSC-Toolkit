"""
Tests for audio processors and caching
"""
import pytest
import torch
import pathlib
from rsc_toolkit.processors.base import BaseAudioProcessor

# Try to import MelBankAudioProcessor, skip tests if not available
try:
    from rsc_toolkit.processors.mel import MelBankAudioProcessor
    HAS_MEL = True
except ImportError:
    HAS_MEL = False


class DummyProcessor(BaseAudioProcessor):
    """Dummy processor for testing"""
    param1: int = 10
    param2: str = "test"
    
    def _get_processor_config(self):
        return {
            "param1": self.param1,
            "param2": self.param2,
        }
    
    def make_features(self, audio: torch.Tensor, sample_id=None):
        # Simple dummy feature extraction
        return audio.mean(dim=-1, keepdim=True)


def test_processor_without_caching():
    """Test processor works without caching"""
    processor = DummyProcessor(param1=20, param2="hello")
    
    audio = torch.randn(1, 16000)  # 1 second at 16kHz
    features = processor.make_features(audio)
    
    assert features is not None
    assert features.shape == (1, 1)


def test_processor_caching_enabled(tmp_path):
    """Test processor with caching enabled"""
    cache_dir = str(tmp_path)
    
    processor = DummyProcessor(
        param1=20,
        param2="hello",
        cache_features=True,
        cache_dir=cache_dir
    )
    
    # Cache should be initialized
    assert processor._feature_cache is not None
    
    sample_id = "test_sample_1"
    audio = torch.randn(1, 16000)
    
    # First call - should compute and cache
    features1 = processor.make_features_cached(audio, sample_id)
    
    # Second call - should use cache
    features2 = processor.make_features_cached(audio, sample_id)
    
    # Should be the same
    assert torch.allclose(features1, features2)
    
    # Verify it's actually cached
    assert processor._feature_cache.is_cached(sample_id)


def test_processor_cache_isolation(tmp_path):
    """Test that different processor configs use different caches"""
    cache_dir = str(tmp_path)
    
    # Processor with param1=10
    proc1 = DummyProcessor(
        param1=10,
        cache_features=True,
        cache_dir=cache_dir
    )
    
    # Processor with param1=20
    proc2 = DummyProcessor(
        param1=20,
        cache_features=True,
        cache_dir=cache_dir
    )
    
    sample_id = "shared_sample"
    audio = torch.randn(1, 16000)
    
    # Process with both
    features1 = proc1.make_features_cached(audio, sample_id)
    features2 = proc2.make_features_cached(audio, sample_id)
    
    # They should have different cache directories
    assert proc1._feature_cache.cache_key != proc2._feature_cache.cache_key
    assert proc1._feature_cache.features_dir != proc2._feature_cache.features_dir
    
    # Both should be cached in their respective caches
    assert proc1._feature_cache.is_cached(sample_id)
    assert proc2._feature_cache.is_cached(sample_id)


@pytest.mark.skipif(not HAS_MEL, reason="MelBankAudioProcessor requires torchaudio")
def test_melbank_processor(tmp_path):
    """Test MelBankAudioProcessor"""
    processor = MelBankAudioProcessor(
        n_mels=64,
        sampling_rate=16000,
        cache_features=True,
        cache_dir=str(tmp_path)
    )
    
    # Audio should be 16kHz for fbank
    audio = torch.randn(1, 16000)
    
    # Should work without caching
    features = processor.make_features(audio)
    assert features is not None
    
    # Should work with caching
    sample_id = "mel_sample_1"
    cached_features = processor.make_features_cached(audio, sample_id)
    assert cached_features is not None
    assert processor._feature_cache.is_cached(sample_id)


def test_processor_config_extraction():
    """Test that processor config excludes cache-related fields"""
    processor = DummyProcessor(
        param1=10,
        param2="test",
        cache_features=True,
        cache_dir="/tmp/test"
    )
    
    config = processor._get_processor_config()
    
    # Should include processor-specific params
    assert "param1" in config
    assert "param2" in config
    
    # Should NOT include cache-related fields
    assert "cache_features" not in config
    assert "cache_dir" not in config
    assert "_feature_cache" not in config


def test_processor_cache_key_changes_with_config(tmp_path):
    """Test that cache key changes when processor config changes"""
    cache_dir = str(tmp_path)
    
    # Processor with param1=10
    proc1 = DummyProcessor(
        param1=10,
        cache_features=True,
        cache_dir=cache_dir
    )
    
    # Processor with param1=20 (different config)
    proc2 = DummyProcessor(
        param1=20,
        cache_features=True,
        cache_dir=cache_dir
    )
    
    # Cache keys should be different
    assert proc1._feature_cache.cache_key != proc2._feature_cache.cache_key

