"""
Tests for PyTorch DataLoader integration
"""
import pytest
import pathlib
import torch
from torch.utils.data import DataLoader
from rsc_toolkit.datasets.toy_dataset import ToyDataset
from rsc_toolkit.datasets.loaders import ICBHIPyTorchDataset, create_dataloader
from rsc_toolkit.processors.base import BaseAudioProcessor
from typing import Optional, Dict, Any


class SimpleTestProcessor(BaseAudioProcessor):
    """Simple processor for testing"""
    feature_dim: int = 128
    
    def _get_processor_config(self) -> Dict[str, Any]:
        return {"feature_dim": self.feature_dim}
    
    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        features = audio.mean(dim=-1, keepdim=True)
        repeat_factor = (self.feature_dim // features.shape[-1]) + 1
        features = features.repeat(1, repeat_factor)
        return features[:, :self.feature_dim]


@pytest.fixture
def toy_dataset_path():
    """Get path to toy dataset"""
    project_root = pathlib.Path(__file__).parent.parent.parent
    toy_path = project_root / "sample_data" / "toy_dataset" / "files"
    if not toy_path.exists():
        pytest.skip(f"Toy dataset not found at {toy_path}")
    return toy_path


@pytest.fixture
def processed_toy_dataset(tmp_path, toy_dataset_path):
    """Process toy dataset and return processor with cycles"""
    cache_dir = str(tmp_path / "cache")
    
    processor = ToyDataset.make_processor(
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
    
    dataset = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor
    )
    
    # Process to get cycles
    dataset.process()
    
    # Get cycles and recordings from processor
    all_cycles = processor.get_cycles()
    recordings = processor.get_recordings()
    
    return {
        'processor': processor,
        'cycles': all_cycles,
        'recordings': recordings
    }


@pytest.mark.slow
def test_pytorch_dataset_cycle_level(processed_toy_dataset):
    """Test PyTorch Dataset with cycle-level samples"""
    cycles = processed_toy_dataset['cycles']
    processor = processed_toy_dataset['processor']
    
    # Create dataset
    dataset = ICBHIPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor.feature_processor,
        use_cached_features=True
    )
    
    assert len(dataset) > 0
    assert len(dataset) == len(cycles)
    
    # Get a sample
    features, label = dataset[0]
    
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 128  # feature_dim
    assert isinstance(label, (int, torch.Tensor))
    
    # Check that features are cached
    sample_id = dataset._get_sample_id(cycles[0])
    assert processor.feature_processor._feature_cache.is_cached(sample_id)


@pytest.mark.slow
def test_pytorch_dataset_clip_level(processed_toy_dataset):
    """Test PyTorch Dataset with clip-level (recording) samples"""
    recordings = processed_toy_dataset['recordings']
    processor = processed_toy_dataset['processor']
    
    # Create dataset
    dataset = ICBHIPyTorchDataset(
        samples=recordings,
        sample_level="clip",
        feature_processor=processor.feature_processor,
        use_cached_features=True
    )
    
    assert len(dataset) > 0
    assert len(dataset) == len(recordings)
    
    # Get a sample
    features, label = dataset[0]
    
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, (int, torch.Tensor))


@pytest.mark.slow
def test_dataloader_basic(processed_toy_dataset):
    """Test creating and using a DataLoader"""
    cycles = processed_toy_dataset['cycles']
    processor = processed_toy_dataset['processor']
    
    dataset = ICBHIPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor.feature_processor,
        use_cached_features=True
    )
    
    # Create DataLoader
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # Get a batch
    batch_features, batch_labels = next(iter(dataloader))
    
    assert isinstance(batch_features, torch.Tensor)
    assert isinstance(batch_labels, torch.Tensor)
    assert batch_features.shape[0] == 4  # batch_size
    assert batch_labels.shape[0] == 4
    assert batch_features.shape[1] == 128  # feature_dim


@pytest.mark.slow
def test_dataloader_multiple_batches(processed_toy_dataset):
    """Test iterating through multiple batches"""
    cycles = processed_toy_dataset['cycles']
    processor = processed_toy_dataset['processor']
    
    dataset = ICBHIPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor.feature_processor,
        use_cached_features=True
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,  # Don't shuffle for predictable iteration
        num_workers=0
    )
    
    # Iterate through a few batches
    batch_count = 0
    for batch_features, batch_labels in dataloader:
        assert batch_features.shape[0] <= 4
        assert batch_labels.shape[0] == batch_features.shape[0]
        batch_count += 1
        if batch_count >= 3:  # Just test first 3 batches
            break
    
    assert batch_count > 0


@pytest.mark.slow
def test_dataloader_with_transform(processed_toy_dataset):
    """Test DataLoader with transforms"""
    cycles = processed_toy_dataset['cycles']
    processor = processed_toy_dataset['processor']
    
    # Define a simple transform
    def feature_transform(features):
        return features / features.max()  # Normalize
    
    dataset = ICBHIPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor.feature_processor,
        use_cached_features=True,
        transform=feature_transform
    )
    
    dataloader = create_dataloader(dataset, batch_size=4, num_workers=0)
    batch_features, batch_labels = next(iter(dataloader))
    
    # Check that transform was applied (max should be <= 1.0)
    assert batch_features.max() <= 1.0 + 1e-6  # Allow small floating point errors


@pytest.mark.slow
def test_dataloader_label_extraction(processed_toy_dataset):
    """Test that labels are correctly extracted from cycles"""
    cycles = processed_toy_dataset['cycles']
    processor = processed_toy_dataset['processor']
    
    dataset = ICBHIPyTorchDataset(
        samples=cycles,
        sample_level="cycle",
        feature_processor=processor.feature_processor,
        use_cached_features=True
    )
    
    # Check labels are valid
    labels = []
    for i in range(min(10, len(dataset))):
        _, label = dataset[i]
        labels.append(int(label))
        assert 0 <= label <= 3  # ICBHI has 4 classes: Normal, Crackle, Wheeze, Both
    
    # Should have some variety (not all same label)
    assert len(set(labels)) > 0

