"""
Unit tests for PyTorch Dataset loaders with mocks.
These tests don't require actual audio files or slow processing.
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from rsc_toolkit.datasets.loaders import (
    BaseRSCPyTorchDataset,
    ICBHIPyTorchDataset,
    create_dataloader
)
from rsc_toolkit.processors.base import BaseAudioProcessor
from typing import Optional, Dict, Any


class MockCycle:
    """Mock Cycle object"""
    def __init__(self, audio, start, end, annotation=None, label=None):
        self.audio = audio
        self.start = start
        self.end = end
        self.annotation = annotation or {}
        self.label = label
        self._recording = None


class MockRecording:
    """Mock Recording object"""
    def __init__(self, wav_path, meta=None):
        self.wav_path = wav_path
        self.meta = meta or Mock()


class MockProcessor(BaseAudioProcessor):
    """Mock feature processor"""
    feature_dim: int = 128
    _feature_cache: Optional[Any] = None
    
    def _get_processor_config(self) -> Dict[str, Any]:
        return {"feature_dim": self.feature_dim}
    
    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        # Simple mock: just return a tensor of the right shape
        return torch.randn(self.feature_dim)
    
    def make_features_cached(self, audio: torch.Tensor, sample_id: str):
        # Mock cached version
        if self._feature_cache and self._feature_cache.is_cached(sample_id):
            return self._feature_cache.get(sample_id)
        features = self.make_features(audio, sample_id)
        if self._feature_cache:
            self._feature_cache.put(sample_id, features)
        return features


class MockFeatureCache:
    """Mock feature cache"""
    def __init__(self):
        self._cache = {}
    
    def is_cached(self, sample_id: str) -> bool:
        return sample_id in self._cache
    
    def get(self, sample_id: str) -> torch.Tensor:
        return self._cache[sample_id]
    
    def put(self, sample_id: str, features: torch.Tensor):
        self._cache[sample_id] = features


def test_base_dataset_cycle_level():
    """Test BaseRSCPyTorchDataset with cycle-level samples"""
    # Create mock cycles
    cycles = [
        MockCycle(
            audio=torch.randn(1, 1000),
            start=0.0,
            end=1.0,
            annotation={'Crackle': 0, 'Wheeze': 0},
            label=0
        ),
        MockCycle(
            audio=torch.randn(1, 1000),
            start=1.0,
            end=2.0,
            annotation={'Crackle': 1, 'Wheeze': 0},
            label=1
        ),
    ]
    
    processor = MockProcessor()
    processor._feature_cache = MockFeatureCache()
    
    dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor,
        use_cached_features=True
    )
    
    assert len(dataset) == 2
    assert dataset.sample_level == "cycle"
    
    # Get a sample
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (128,)  # feature_dim
    assert label == 0


def test_base_dataset_clip_level():
    """Test BaseRSCPyTorchDataset with clip-level samples"""
    # For unit tests, we'll skip clip-level with file loading
    # That's tested in slow integration tests
    # Here we just verify the filtering works
    recordings = [
        MockRecording(wav_path=Mock(stem="rec1")),
        MockRecording(wav_path=Mock(stem="rec2")),
    ]
    
    processor = MockProcessor()
    
    dataset = BaseRSCPyTorchDataset(
        samples=recordings,
        sample_level="clip",
        feature_processor=processor,
        use_cached_features=False
    )
    
    assert len(dataset) == 2
    assert dataset.sample_level == "clip"
    # Note: Actually loading from wav_path is tested in slow tests


def test_dataset_filters_by_sample_level():
    """Test that dataset filters samples by sample_level"""
    # Mix of cycles and recordings
    samples = [
        MockCycle(audio=torch.randn(1, 100), start=0.0, end=1.0),
        MockRecording(wav_path=Mock(stem="rec1")),
        MockCycle(audio=torch.randn(1, 100), start=1.0, end=2.0),
    ]
    
    processor = MockProcessor()
    
    # Cycle level should only include cycles
    cycle_dataset = BaseRSCPyTorchDataset(
        samples=samples,
        sample_level="cycle",
        feature_processor=processor
    )
    assert len(cycle_dataset) == 2
    
    # Clip level should only include recordings (filtering works)
    clip_dataset = BaseRSCPyTorchDataset(
        samples=samples,
        sample_level="clip",
        feature_processor=processor
    )
    assert len(clip_dataset) == 1
    # Note: Actually loading from files is tested in slow tests


def test_dataset_label_extraction():
    """Test label extraction from samples"""
    cycles = [
        MockCycle(
            audio=torch.randn(1, 100),
            start=0.0,
            end=1.0,
            annotation={'Crackle': 0, 'Wheeze': 0},
            label=None  # Should use annotation
        ),
        MockCycle(
            audio=torch.randn(1, 100),
            start=1.0,
            end=2.0,
            annotation={'Crackle': 1, 'Wheeze': 0},
            label=None
        ),
        MockCycle(
            audio=torch.randn(1, 100),
            start=2.0,
            end=3.0,
            label=3  # Explicit label
        ),
    ]
    
    processor = MockProcessor()
    
    dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor
    )
    
    # First: Normal (0, 0) -> 0
    _, label1 = dataset[0]
    assert label1 == 0
    
    # Second: Crackle (1, 0) -> 1
    _, label2 = dataset[1]
    assert label2 == 1
    
    # Third: Explicit label -> 3
    _, label3 = dataset[2]
    assert label3 == 3


def test_dataset_custom_label_function():
    """Test custom label extraction function"""
    cycles = [
        MockCycle(audio=torch.randn(1, 100), start=0.0, end=1.0, label=5),
    ]
    
    processor = MockProcessor()
    
    def custom_label_fn(sample):
        return sample.label * 2  # Custom transformation
    
    dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor,
        get_label_fn=custom_label_fn
    )
    
    _, label = dataset[0]
    assert label == 10  # 5 * 2


def test_dataset_with_transform():
    """Test dataset with feature transform"""
    cycles = [
        MockCycle(audio=torch.randn(1, 100), start=0.0, end=1.0),
    ]
    
    processor = MockProcessor()
    
    def transform(features):
        return features * 2
    
    dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor,
        transform=transform
    )
    
    features, _ = dataset[0]
    # Features should be transformed (multiplied by 2)
    # Since processor returns randn, we can't check exact value, but shape should be right
    assert features.shape == (128,)


def test_dataset_cached_features():
    """Test that cached features are used when available"""
    cycles = [
        MockCycle(audio=torch.randn(1, 100), start=0.0, end=1.0),
    ]
    
    processor = MockProcessor()
    cache = MockFeatureCache()
    processor._feature_cache = cache
    
    # Pre-cache a feature
    # Use dataset's sample_id convention to ensure cache hit
    temp_dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor,
        use_cached_features=True
    )
    sample_id = temp_dataset._get_sample_id(cycles[0])
    cached_feature = torch.ones(128) * 42.0  # Distinct value
    cache.put(sample_id, cached_feature)
    
    dataset = temp_dataset
    
    features, _ = dataset[0]
    # Should use cached feature
    assert torch.allclose(features, cached_feature)


def test_create_dataloader():
    """Test create_dataloader helper function"""
    cycles = [
        MockCycle(audio=torch.randn(1, 100), start=i*1.0, end=(i+1)*1.0)
        for i in range(10)
    ]
    
    processor = MockProcessor()
    
    dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    # Get a batch
    batch_features, batch_labels = next(iter(dataloader))
    
    assert batch_features.shape == (4, 128)  # batch_size, feature_dim
    assert batch_labels.shape == (4,)


def test_dataloader_shuffle():
    """Test that DataLoader shuffles when requested"""
    cycles = [
        MockCycle(audio=torch.randn(1, 100), start=i*1.0, end=(i+1)*1.0, label=i)
        for i in range(10)
    ]
    
    processor = MockProcessor()
    
    dataset = BaseRSCPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor
    )
    
    # Get labels in order
    ordered_labels = [dataset[i][1] for i in range(10)]
    
    # Create shuffled dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0
    )
    
    # Get all labels from dataloader
    batch_features, batch_labels = next(iter(dataloader))
    shuffled_labels = batch_labels.tolist()
    
    # Should be different order (with high probability)
    # Note: There's a small chance they're the same, but very unlikely
    assert shuffled_labels != ordered_labels or len(set(shuffled_labels)) == 1


def test_icbhi_dataset_class():
    """Test ICBHIPyTorchDataset is a valid subclass"""
    cycles = [
        MockCycle(audio=torch.randn(1, 100), start=0.0, end=1.0),
    ]
    
    processor = MockProcessor()
    
    dataset = ICBHIPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor
    )
    
    assert isinstance(dataset, BaseRSCPyTorchDataset)
    assert len(dataset) == 1
