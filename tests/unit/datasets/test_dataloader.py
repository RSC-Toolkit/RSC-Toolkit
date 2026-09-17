# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Tests for DataLoader integration driven by the CSV sample index.
"""
import pytest
import pathlib
import torch
from typing import Optional, Dict, Any

from rsc_toolkit.datasets.toy_dataset import ToyDataset
from rsc_toolkit.datasets.index_dataset import IndexDataset
from rsc_toolkit.processors.base import BaseAudioProcessor
from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor


class SimpleTestProcessor(BaseAudioProcessor):
    """Simple processor for testing"""

    def __init__(self, feature_dim: int = 128, **kwargs):
        self.feature_dim = feature_dim
        super().__init__(**kwargs)
    
    def get_processor_config(self) -> Dict[str, Any]:
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
    project_root = pathlib.Path(__file__).parent.parent.parent.parent
    return project_root / "sample_data" / "toy_dataset" / "files"


@pytest.fixture
def processor_with_index(tmp_path, toy_dataset_path):
    """Build index CSVs and return processor and index paths."""
    cache_dir = str(tmp_path / "cache")

    processor = ToyDataset.make_processor(
        outputs_directory=cache_dir,
        raw_data_directory=str(toy_dataset_path),
        preprocess_features=False,
        force=True,
    )

    dataset = ToyDataset(
        downloader=ToyDataset.make_downloader(outputs_directory=cache_dir),
        processor=processor,
    )

    dataset.download()
    cycle_index = processor.build_index(sample_level="cycle")
    clip_index = processor.build_index(sample_level="clip")

    return {
        "processor": processor,
        "cycle_index_path": processor.index_dir_path / "index_cycle.csv",
        "clip_index_path": processor.index_dir_path / "index_clip.csv",
    }


@pytest.mark.slow
def test_cycle_level_index_dataloader(processor_with_index, toy_dataset_path):
    """Cycle-level dataset reading from CSV index."""
    processor = processor_with_index["processor"]
    index_path = processor_with_index["cycle_index_path"]

    ds = IndexDataset(
        index_csv=index_path,
        data_root=toy_dataset_path,
        sampling_rate=processor.sampling_rate,
        sample_level="cycle",
    )
    item = ds[0]
    audio = item["audio"]
    label = item["label"]
    sample_id = item["sample_id"]

    assert isinstance(audio, torch.Tensor)
    assert isinstance(label, int)
    assert 0 <= label <= 3
    assert isinstance(sample_id, str)

    fp = SimpleTestProcessor(feature_dim=128)
    feats = fp.make_features(audio)
    # Expect feature_dim on last axis (C, feature_dim) when C=1 -> (1,128)
    assert feats.shape[-1] == fp.feature_dim
    assert feats.squeeze(0).shape[0] == fp.feature_dim


@pytest.mark.slow
def test_clip_level_index_dataloader(processor_with_index, toy_dataset_path):
    """Clip-level dataset reading from CSV index."""
    processor = processor_with_index["processor"]
    index_path = processor_with_index["clip_index_path"]

    ds = IndexDataset(
        index_csv=index_path,
        data_root=toy_dataset_path,
        sampling_rate=processor.sampling_rate,
        sample_level="clip",
    )
    item = ds[0]
    audio = item["audio"]
    label = item["label"]

    assert isinstance(audio, torch.Tensor)
    # Clip-level label may be None; that's acceptable for clip indexes.
    assert label is None or isinstance(label, int)


@pytest.mark.slow
def test_dataloader_basic(processor_with_index, toy_dataset_path):
    """Basic DataLoader over cycle-level index."""
    processor = processor_with_index["processor"]
    index_path = processor_with_index["cycle_index_path"]

    ds = IndexDataset(
        index_csv=index_path,
        data_root=toy_dataset_path,
        sampling_rate=processor.sampling_rate,
        sample_level="cycle",
    )
    items = [ds[i] for i in range(min(4, len(ds)))]
    assert len(items) > 0
    for item in items:
        assert isinstance(item["audio"], torch.Tensor)
        assert isinstance(item["label"], int)


@pytest.mark.slow
def test_dataloader_multiple_batches(processor_with_index, toy_dataset_path):
    """Iterate through multiple batches from index-backed dataset."""
    processor = processor_with_index["processor"]
    index_path = processor_with_index["cycle_index_path"]

    ds = IndexDataset(
        index_csv=index_path,
        data_root=toy_dataset_path,
        sampling_rate=processor.sampling_rate,
        sample_level="cycle",
    )
    count = 0
    for i in range(min(10, len(ds))):
        item = ds[i]
        assert isinstance(item["audio"], torch.Tensor)
        assert isinstance(item["label"], int)
        count += 1
    assert count > 0


# @pytest.mark.slow # does not make sense
# def test_dataloader_with_transform(processor_with_index, toy_dataset_path):
#     """Test transform applied after feature extraction (manual in test)."""
#     processor = processor_with_index["processor"]
#     index_path = processor_with_index["cycle_index_path"]

#     ds = IndexDataset(
#         index_csv=index_path,
#         data_root=toy_dataset_path,
#         sampling_rate=processor.sampling_rate,
#         sample_level="cycle",
#     )
#     item = ds[0]
#     audio = item["audio"]

#     fp = SimpleTestProcessor(feature_dim=128)
#     feats = fp.make_features(audio)
#     normed = feats / feats.max()
#     assert normed.max() <= 1.0 + 1e-6


@pytest.mark.slow
def test_dataloader_label_extraction(processor_with_index, toy_dataset_path):
    """Labels from index-backed dataset are in expected range."""
    processor = processor_with_index["processor"]
    index_path = processor_with_index["cycle_index_path"]

    ds = IndexDataset(
        index_csv=index_path,
        data_root=toy_dataset_path,
        sampling_rate=processor.sampling_rate,
        sample_level="cycle",
    )

    labels = []
    for i in range(min(10, len(ds))):
        item = ds[i]
        labels.append(int(item["label"]))
        assert 0 <= item["label"] <= 3

    assert len(set(labels)) > 0


@pytest.mark.slow
def test_cycle_level_index_dataloader_with_audio_preprocessor(processor_with_index, toy_dataset_path):
    """Audio preprocessor is applied to cycle slices."""
    processor = processor_with_index["processor"]
    index_path = processor_with_index["cycle_index_path"]

    cfg = AudioPreprocessConfig(desired_length_sec=1.0, pad_type="repeat")
    pre = AudioPreprocessor(sampling_rate=processor.sampling_rate, cfg=cfg)

    ds = IndexDataset(
        index_csv=index_path,
        data_root=toy_dataset_path,
        sampling_rate=processor.sampling_rate,
        sample_level="cycle",
        audio_preprocessor=pre,
    )
    item = ds[0]
    audio = item["audio"]
    assert audio.shape[1] == processor.sampling_rate
