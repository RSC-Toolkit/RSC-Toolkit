"""
Tests for audio processors and caching
"""
import pytest
import torch

from rsc_toolkit.datasets.feature_cache import FeatureCache
from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor
from rsc_toolkit.processors.base import BaseAudioProcessor
from rsc_toolkit.processors.mel import MelBankAudioProcessor


class DummyProcessor(BaseAudioProcessor):
    """Dummy processor for testing"""

    def __init__(
        self,
        param1: int = 10,
        param2: str = "test",
        feature_cache = None,
        **kwargs,
        ):
        self.param1 = param1
        self.param2 = param2
        super().__init__(feature_cache=feature_cache, **kwargs)
    
    def get_processor_config(self):
        return {
            "param1": self.param1,
            "param2": self.param2,
        }
    
    def make_features(self, audio: torch.Tensor, sample_id=None):
        # Simple dummy feature extraction
        return audio.mean(dim=-1, keepdim=True)


@pytest.fixture
def feature_cache(tmp_path):
    return FeatureCache(cache_dir=tmp_path)
    

def test_processor_without_caching(feature_cache):
    """Test processor works without caching"""
    processor = DummyProcessor(param1=20, param2="hello", feature_cache=feature_cache)
    feature_cache.configure(
        processor_config=processor.get_processor_config(),
        processor_name=processor.__class__.__name__
    )
    
    audio = torch.randn(1, 16000)  # 1 second at 16kHz
    features = processor.make_features(audio)
    
    assert features is not None
    assert features.shape == (1, 1)


def test_processor_caching_enabled(feature_cache, tmp_path):
    """Test processor with caching enabled"""

    processor = DummyProcessor(
        param1=20,
        param2="hello",
        feature_cache=feature_cache,
    )
    feature_cache.configure(
        processor_config=processor.get_processor_config(),
        processor_name=processor.__class__.__name__
    )
    
    # Cache should be initialized
    assert processor.feature_cache is not None
    
    sample_id = "test_sample_1"
    audio = torch.randn(1, 16000)
    
    # First call - should compute and cache
    features1 = processor.make_features_cached(audio, sample_id)
    
    # Second call - should use cache
    features2 = processor.make_features_cached(audio, sample_id)
    
    # Should be the same
    assert torch.allclose(features1, features2)
    
    # Verify it's actually cached
    assert processor.feature_cache.is_cached(sample_id)


def test_processor_cache_isolation(tmp_path):
    """Test that different processor configs use different caches"""

    # Processor with param1=10
    proc1 = DummyProcessor(
        param1=10,
        feature_cache=FeatureCache(cache_dir=tmp_path / "1"),
    )
    proc1.feature_cache.configure(
        processor_config=proc1.get_processor_config(),
        processor_name=proc1.__class__.__name__
    )
    
    # Processor with param1=20
    proc2 = DummyProcessor(
        param1=20,
        feature_cache=FeatureCache(cache_dir=tmp_path/ "2"),
    )
    proc2.feature_cache.configure(
        processor_config=proc2.get_processor_config(),
        processor_name=proc2.__class__.__name__
    )
    
    sample_id = "shared_sample"
    audio = torch.randn(1, 16000)
    
    # Process with both
    features1 = proc1.make_features_cached(audio, sample_id)
    features2 = proc2.make_features_cached(audio, sample_id)
    
    # They should have different cache directories
    assert proc1.feature_cache.cache_key != proc2.feature_cache.cache_key
    assert proc1.feature_cache.features_dir != proc2.feature_cache.features_dir
    
    # Both should be cached in their respective caches
    assert proc1.feature_cache.is_cached(sample_id)
    assert proc2.feature_cache.is_cached(sample_id)


def test_melbank_processor(tmp_path, feature_cache):
    """Test MelBankAudioProcessor"""
    processor = MelBankAudioProcessor(
        n_mels=64,
        sampling_rate=16000,
        target_frames=64,
        feature_cache=feature_cache,
    )
    processor.feature_cache.configure(
        processor_config=processor.get_processor_config(),
        processor_name=processor.__class__.__name__
    )
    
    # Audio should be 16kHz for fbank
    audio = torch.randn(1, 16000)
    
    # Should work without caching
    features = processor.make_features(audio)
    assert features is not None
    assert features.shape == (64, 64)
    
    # Should work with caching
    sample_id = "mel_sample_1"
    cached_features = processor.make_features_cached(audio, sample_id)
    assert cached_features is not None
    assert processor.feature_cache.is_cached(sample_id)


def test_processor_config_extraction(feature_cache):
    """Test that processor config excludes cache-related fields"""
    processor = DummyProcessor(
        param1=10,
        param2="test",
        feature_cache=feature_cache,
    )
    
    config = processor.get_processor_config()
    
    # Should include processor-specific params
    assert "param1" in config
    assert "param2" in config
    
    # Should NOT include cache-related fields
    assert "cache_features" not in config
    assert "cache_dir" not in config
    assert "feature_cache" not in config


def test_processor_cache_key_changes_with_config(tmp_path):
    """Test that cache key changes when processor config changes"""

    # Processor with param1=10
    proc1 = DummyProcessor(
        param1=10,
        feature_cache=FeatureCache(cache_dir=tmp_path / "1"),
    )
    proc1.feature_cache.configure(
        processor_config=proc1.get_processor_config(),
        processor_name=proc1.__class__.__name__
    )
    
    # Processor with param1=20
    proc2 = DummyProcessor(
        param1=20,
        feature_cache=FeatureCache(cache_dir=tmp_path/ "2"),
    )
    proc2.feature_cache.configure(
        processor_config=proc2.get_processor_config(),
        processor_name=proc2.__class__.__name__
    )
    # Cache keys should be different
    assert proc1.feature_cache.cache_key != proc2.feature_cache.cache_key


def test_audio_preprocessor_repeat_pad():
    cfg = AudioPreprocessConfig(desired_length_sec=1.0, pad_type="repeat")
    pre = AudioPreprocessor(sampling_rate=8, cfg=cfg)
    audio = torch.tensor([[0.0, 1.0, 2.0]])
    out = pre(audio)
    assert out.shape == (1, 8)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]]))


def test_audio_preprocessor_zero_pad():
    cfg = AudioPreprocessConfig(desired_length_sec=1.0, pad_type="zero")
    pre = AudioPreprocessor(sampling_rate=5, cfg=cfg)
    audio = torch.tensor([[1.0, 2.0]])
    out = pre(audio)
    assert out.shape == (1, 5)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]]))


def test_audio_preprocessor_trim():
    cfg = AudioPreprocessConfig(desired_length_sec=1.0, pad_type="repeat")
    pre = AudioPreprocessor(sampling_rate=3, cfg=cfg)
    audio = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = pre(audio)
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]))
