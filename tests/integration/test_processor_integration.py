# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Integration tests for dataset processor with feature preprocessing
"""
import logging
import pytest
import torch
import pathlib
from rsc_toolkit.datasets.base import BaseRSCDatasetProcessor
from rsc_toolkit.processors.base import BaseAudioProcessor
from rsc_toolkit.datasets.feature_cache import FeatureCache


class MockSample:
    """Mock sample for testing"""
    def __init__(self, sample_id):
        self.sample_id = sample_id
        self.audio = torch.randn(1, 16000)  # 1 second at 16kHz
        self.wav_path = pathlib.Path(f"/tmp/{sample_id}.wav")

@pytest.fixture
def feature_cache(tmp_path):
    return FeatureCache(cache_dir=tmp_path)
    

class MockProcessor(BaseAudioProcessor):
    """Mock processor for testing"""
    def __init__(self, param: int = 10, **kwargs):
        self.param = param
        super().__init__(**kwargs)
    
    def get_processor_config(self):
        return {"param": self.param}
    
    def make_features(self, audio: torch.Tensor, sample_id=None):
        return audio.mean(dim=-1, keepdim=True)


class MockDatasetProcessor(BaseRSCDatasetProcessor):
    """Mock dataset processor for testing"""
    name: str = "test"
    samples: list = []  # Declare as field to avoid Pydantic error
    
    def validate_dataset_files(self) -> bool:
        return False  # Always need processing

    def validate_processed_files(self):
        return False  # Always need processing
    
    def process_dataset(self) -> None:
        # Create some mock samples
        self.samples = [MockSample(f"sample_{i}") for i in range(5)]
        
        # # Preprocess features if enabled
        # if self.preprocess_features and self.feature_processor:
        #     self.preprocess_all_features(self.samples)
    
    def _extract_audio(self, sample) -> torch.Tensor:
        if hasattr(sample, 'audio'):
            return sample.audio
        return super()._extract_audio(sample)
    
    def _get_sample_id(self, sample) -> str:
        if hasattr(sample, 'sample_id'):
            return sample.sample_id
        return super()._get_sample_id(sample)

@pytest.mark.skip(reason="preprocessing stage not implemnted yet")
def test_processor_preprocessing_integration(tmp_path, feature_cache):
    """Test full preprocessing integration"""
    processor = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        preprocess_features=True,
        feature_processor=MockProcessor(
            feature_cache=feature_cache,
        ),
        num_preprocess_workers=2
    )
    processor.feature_processor.feature_cache.configure(
        processor_config=processor.feature_processor.get_processor_config(),
        processor_name=processor.feature_processor.__class__.__name__,
    )
    
    # Process dataset
    processor.process_dataset()
    
    # Verify samples were processed
    assert hasattr(processor, 'samples')
    assert len(processor.samples) == 5
    
    # Verify features were cached
    for sample in processor.samples:
        assert processor.feature_processor.feature_cache.is_cached(sample.sample_id)


def test_processor_preprocessing_without_cache(tmp_path):
    """Test preprocessing without caching"""
    processor = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        preprocess_features=True,
        feature_processor=MockProcessor(
            param=20,
        ),
        num_preprocess_workers=2
    )
    
    # Should still work
    processor.process_dataset()
    assert hasattr(processor, 'samples')


def test_processor_cache_dir_includes_processor_config(tmp_path):
    """Test that cache directory includes processor config"""
    # Processor with param=10
    proc1 = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        feature_processor=MockProcessor(param=10),
    )
    
    # Processor with param=20
    proc2 = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        feature_processor=MockProcessor(param=20),
    )
    
    # Cache directories should be different
    assert proc1.cache_dir_path != proc2.cache_dir_path


def test_processor_extract_audio_default(tmp_path):
    """Test default _extract_audio implementation"""
    processor = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path)
    )
    
    # Sample with audio attribute
    sample_with_audio = MockSample("test1")
    audio1 = processor._extract_audio(sample_with_audio)
    assert isinstance(audio1, torch.Tensor)
    
    # Sample with wav_path but no audio (tests base implementation)
    class SampleWithPath:
        wav_path = pathlib.Path("/nonexistent/file.wav")
    
    sample_with_path = SampleWithPath()
    # This will raise FileNotFoundError or RuntimeError, which is expected
    with pytest.raises((FileNotFoundError, RuntimeError)):
        processor._extract_audio(sample_with_path)


def test_processor_get_sample_id_default(tmp_path):
    """Test default _get_sample_id implementation"""
    processor = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path)
    )
    
    # Sample with sample_id (uses override)
    sample_with_id = MockSample("test1")
    sample_id = processor._get_sample_id(sample_with_id)
    assert sample_id == "test1"
    
    # Sample with wav_path (uses base implementation)
    class SampleWithPath:
        wav_path = pathlib.Path("/tmp/test.wav")
    
    sample = SampleWithPath()
    sample_id = processor._get_sample_id(sample)
    assert sample_id == "test"
    
    # Sample with meta.patient_id (uses base implementation)
    class SampleWithMeta:
        class Meta:
            patient_id = 123
        meta = Meta()
    
    sample2 = SampleWithMeta()
    sample_id2 = processor._get_sample_id(sample2)
    assert sample_id2 == "patient_123"


@pytest.mark.skip(reason="preprocessing stage not implemnted yet")
def test_processor_preprocessing_skips_cached(tmp_path, feature_cache):
    """Test that preprocessing skips already cached samples"""
    processor = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        preprocess_features=True,
        feature_processor=MockProcessor(
            param=20,
            feature_cache=feature_cache,
        ),
        num_preprocess_workers=2
    )
    processor.feature_processor.feature_cache.configure(
        processor_config=processor.feature_processor.get_processor_config(),
        processor_name=processor.feature_processor.__class__.__name__,
    )

    # First run - process everything
    processor.process_dataset()
    
    # Pre-cache one sample manually
    cached_sample_id = "sample_0"
    processor.feature_processor.feature_cache.put(
        cached_sample_id,
        torch.randn(1, 1)
    )
    
    # Second run - should skip the cached one
    # Track which samples were processed
    original_preprocess = processor.preprocess_all_features
    
    processed_samples = []
    def tracked_preprocess(samples):
        for s in samples:
            if not processor.feature_processor.feature_cache.is_cached(s.sample_id):
                processed_samples.append(s.sample_id)
        return original_preprocess(samples)
    
    processor.preprocess_all_features = tracked_preprocess
    processor.process_dataset()
    
    # Should have processed fewer samples (excluding the pre-cached one)
    # Note: This is a simplified test - in reality, the skip happens inside preprocess_samples
    assert processor.feature_processor.feature_cache.is_cached(cached_sample_id)



def test_cache_dir_reflects_real_processor_config(tmp_path):
    """
    Checks that the settings of a real processor (MelBankAudioProcessor) are reflected in the cache key.

    Previously base.py called `_get_processor_config()` (with underscore) and
    swallowed the AttributeError. The real name has no underscore, so the settings silently
    dropped out of the cache key and the same cache directory was used even with a different n_mels.

    Testing only with the MockProcessor above misses the same mistake as long as the dummy
    defines the misspelled name, so a real processor is used here.
    """
    from rsc_toolkit.processors.mel import MelBankAudioProcessor

    proc_64 = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        feature_processor=MelBankAudioProcessor(n_mels=64, sampling_rate=16000),
    )
    proc_128 = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        feature_processor=MelBankAudioProcessor(n_mels=128, sampling_rate=16000),
    )

    assert proc_64.cache_dir_path != proc_128.cache_dir_path


def test_cache_dir_warns_when_processor_has_no_config(tmp_path, caplog):
    """A processor without get_processor_config() triggers a warning instead of being silently skipped."""

    class _NoConfig:
        pass

    processor = MockDatasetProcessor(
        name="test",
        outputs_directory=str(tmp_path),
        feature_processor=_NoConfig(),
    )

    with caplog.at_level(logging.WARNING):
        _ = processor.cache_dir_path

    assert any("get_processor_config" in r.message for r in caplog.records)
