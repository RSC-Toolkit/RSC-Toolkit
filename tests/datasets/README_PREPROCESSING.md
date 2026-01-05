# Testing Toy Dataset Preprocessing

This directory contains tests for the preprocessing and caching system.

## Running Tests

### Run all tests (excluding expensive ones):
```bash
pytest tests/datasets/ -m "not expensive"
```

### Run only expensive preprocessing tests:
```bash
pytest tests/datasets/test_toy_preprocessing.py -m expensive -v
```

### Run a specific expensive test:
```bash
pytest tests/datasets/test_toy_preprocessing.py::test_toy_dataset_preprocessing -m expensive -v
```

## Test Files

- `test_toy_preprocessing.py` - Expensive tests that actually process audio files and cache features
  - Requires the toy dataset at `sample_data/toy_dataset/files/`
  - Tests feature caching, cache reuse, and cache isolation

## What the Tests Do

1. **test_toy_dataset_preprocessing**: 
   - Processes the toy dataset with a simple feature processor
   - Verifies that features are cached correctly
   - Checks cache statistics and directory structure

2. **test_toy_dataset_preprocessing_cache_reuse**:
   - Runs preprocessing twice
   - Verifies that the second run reuses cached features
   - Ensures cache persistence works

3. **test_toy_dataset_different_processors_isolated_cache**:
   - Tests with two different processor configurations
   - Verifies that different processors use isolated caches
   - Ensures cache keys are different for different configs

## Using the Toy Dataset Programmatically

```python
import pathlib
from rsc_toolkit.datasets.toy_dataset import ToyDataset
from rsc_toolkit.processors.base import BaseAudioProcessor
import torch
from typing import Optional, Dict, Any

# Simple test processor
class SimpleTestProcessor(BaseAudioProcessor):
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

# Setup
project_root = pathlib.Path(__file__).parent.parent.parent
toy_path = project_root / "sample_data" / "toy_dataset" / "files"

processor = ToyDataset.make_processor(
    outputs_directory="./test_cache",
    raw_data_directory=str(toy_path),
    preprocess_features=True,
    feature_processor=SimpleTestProcessor(
        feature_dim=128,
        cache_features=True,
        cache_dir="./test_cache"
    ),
    num_preprocess_workers=2
)

dataset = ToyDataset(
    downloader=ToyDataset.make_downloader(outputs_directory="./test_cache"),
    processor=processor
)

# Process
dataset.process()
```

