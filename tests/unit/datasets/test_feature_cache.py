"""
Tests for FeatureCache functionality
"""
import pytest
import logging
import pathlib

import torch

from rsc_toolkit.datasets.feature_cache import FeatureCache


def test_feature_cache_creates_directory(tmp_path):
    """Test that FeatureCache creates cache directory"""
    cache = FeatureCache(
        cache_dir=tmp_path,
    )
    cache.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )
    
    assert cache.features_dir.exists()
    assert cache.features_dir.is_dir()
    assert "features" in str(cache.features_dir)


def test_feature_cache_key_generation():
    """Test that cache keys are deterministic and unique"""
    cache_dir = pathlib.Path("/tmp/test")
    
    cache1 = FeatureCache(
        cache_dir=cache_dir,
    )
    cache1.configure(
        processor_config={"model": "clap", "device": "cuda"},
        processor_name="CLAPProcessor"
    )
    cache2 = FeatureCache(
        cache_dir=cache_dir,
    )
    cache2.configure(
        processor_config={"model": "clap", "device": "cuda"},
        processor_name="CLAPProcessor"
    )
    
    # Same config should give same key
    assert cache1.cache_key == cache2.cache_key
    
    # Different processor name should give different key
    cache3 = FeatureCache(
        cache_dir=cache_dir,
    )
    cache3.configure(
        processor_config={"model": "clap", "device": "cuda"},
        processor_name="MelBankProcessor"
    )
    assert cache1.cache_key != cache3.cache_key
    
    # Different config should give different key
    cache4 = FeatureCache(
        cache_dir=cache_dir,
    )
    cache4.configure(
        processor_config={"model": "clap", "device": "cpu"},
        processor_name="CLAPProcessor"
    )
    
    assert cache1.cache_key != cache4.cache_key


def test_feature_cache_put_get(tmp_path):
    """Test storing and retrieving cached features"""
    cache = FeatureCache(
        cache_dir=tmp_path,
    )
    cache.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )
    
    
    sample_id = "test_sample_123"
    features = torch.randn(128, 512)  # Example feature tensor
    
    # Initially not cached
    assert not cache.is_cached(sample_id)
    assert cache.get(sample_id) is None
    
    # Store features
    cache.put(sample_id, features)
    
    # Should be cached now
    assert cache.is_cached(sample_id)
    
    # Retrieve features
    retrieved = cache.get(sample_id)
    assert retrieved is not None
    assert torch.allclose(retrieved, features)
    assert retrieved.shape == features.shape


def test_feature_cache_metadata_persistence(tmp_path):
    """Test that cache metadata persists across instances"""
    cache_dir = tmp_path
    
    # Create cache and store features
    cache1 = FeatureCache(
        cache_dir=cache_dir,
    )
    cache1.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )
    
    sample_id = "test_sample_456"
    features = torch.randn(64, 256)
    cache1.put(sample_id, features)
    
    # Create new cache instance (simulating restart)
    cache2 = FeatureCache(
        cache_dir=cache_dir,
    )
    cache2.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )
    
    # Should still be cached
    assert cache2.is_cached(sample_id)
    retrieved = cache2.get(sample_id)
    assert torch.allclose(retrieved, features)


def test_feature_cache_multiple_samples(tmp_path):
    """Test caching multiple samples"""
    cache = FeatureCache(
        cache_dir=tmp_path,
    )
    cache.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )
    
    samples = {
        "sample_1": torch.randn(128, 512),
        "sample_2": torch.randn(128, 512),
        "sample_3": torch.randn(128, 512),
    }
    
    # Store all samples
    for sample_id, features in samples.items():
        cache.put(sample_id, features)
    
    # Verify all are cached
    for sample_id, features in samples.items():
        assert cache.is_cached(sample_id)
        retrieved = cache.get(sample_id)
        assert torch.allclose(retrieved, features)
    
    # Check stats
    stats = cache.get_cache_stats()
    assert stats["pt_files"] == 3


def test_feature_cache_isolation(tmp_path):
    """Test that different processors have isolated caches"""
    cache_dir = tmp_path
    
    # CLAP processor cache
    clap_cache = FeatureCache(
        cache_dir=cache_dir,
    )
    clap_cache.configure(
        processor_config={"model_name": "clap", "device": "cuda"},
        processor_name="CLAPProcessor"
    )
    
    # MelBank processor cache
    mel_cache = FeatureCache(
        cache_dir=cache_dir,
    )
    mel_cache.configure(
        processor_config={"n_mels": 128, "sampling_rate": 16000},
        processor_name="MelBankProcessor"
    )
    
    # They should have different cache directories
    assert clap_cache.features_dir != mel_cache.features_dir
    assert clap_cache.cache_key != mel_cache.cache_key
    
    # Store in CLAP cache
    sample_id = "shared_sample"
    clap_features = torch.randn(128, 512)
    clap_cache.put(sample_id, clap_features)
    
    # Should not be in MelBank cache
    assert clap_cache.is_cached(sample_id)
    assert not mel_cache.is_cached(sample_id)
    
    # Store different features in MelBank cache
    mel_features = torch.randn(64, 128)
    mel_cache.put(sample_id, mel_features)
    
    # Both should be cached in their respective caches
    assert clap_cache.is_cached(sample_id)
    assert mel_cache.is_cached(sample_id)
    
    # But features should be different (different shapes)
    clap_retrieved = clap_cache.get(sample_id)
    mel_retrieved = mel_cache.get(sample_id)
    assert clap_retrieved.shape != mel_retrieved.shape
    assert clap_retrieved.shape == (128, 512)
    assert mel_retrieved.shape == (64, 128)


def test_feature_cache_missing_file(tmp_path):
    """
    캐시에 없는 sample_id 를 조회해도 죽지 않는다.

    원래 이 테스트는 `cache.metadata` 와 `cache._save_metadata()` 를 썼는데
    FeatureCache 에는 그런 속성이 없다(스킵 사유 "metadata was agent
    hallucination?" 가 가리키던 것). 현재 구현은 metadata 파일 없이
    디렉토리를 스캔하므로, 실제 동작에 맞춰 다시 썼다.
    """
    cache = FeatureCache(
        cache_dir=tmp_path,
    )
    cache.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )

    sample_id = "missing_sample"

    assert not cache.is_cached(sample_id)
    assert cache.get(sample_id) is None


def test_feature_cache_corrupted_file_returns_none(tmp_path, caplog):
    """
    캐시 파일이 깨져 있으면 예외를 던지지 말고 None 을 돌려준다.

    put() 은 임시 파일에 쓴 뒤 os.replace 로 원자적으로 옮기므로 정상 경로에서는
    깨진 파일이 생기지 않지만, 외부 요인으로 손상된 경우를 방어한다.
    """
    cache = FeatureCache(
        cache_dir=tmp_path,
    )
    cache.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )

    sample_id = "corrupted_sample"
    cache.put(sample_id, torch.randn(8, 8))
    assert cache.is_cached(sample_id)

    # 내용을 망가뜨린다 (파일은 그대로 있으므로 is_cached 는 여전히 True)
    cache._get_cache_path(sample_id).write_bytes(b"not a torch file")

    with caplog.at_level(logging.WARNING):
        assert cache.get(sample_id) is None
    assert any("Failed to load cache" in r.message for r in caplog.records)


def test_feature_cache_stats(tmp_path):
    """Test cache statistics"""
    cache = FeatureCache(
        cache_dir=tmp_path,
    )
    cache.configure(
        processor_config={"test": "config"},
        processor_name="TestProcessor"
    )
    
    # Empty cache
    stats = cache.get_cache_stats()
    assert stats["pt_files"] == 0
    
    # Add some samples
    for i in range(5):
        cache.put(f"sample_{i}", torch.randn(64, 128))
    
    stats = cache.get_cache_stats()
    assert stats["pt_files"] == 5
    assert "cache_dir" in stats

