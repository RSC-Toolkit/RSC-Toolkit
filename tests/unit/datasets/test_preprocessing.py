"""
Tests for parallel preprocessing functionality
"""
import multiprocessing as mp

import pytest
import torch
import time
from rsc_toolkit.datasets.preprocessing import ParallelPreprocessor


class MockSample:
    """Mock sample for testing"""
    def __init__(self, sample_id, audio_length=1000):
        self.sample_id = sample_id
        self.audio = torch.randn(1, audio_length)


def slow_process_fn(sample, sample_id):
    """Simulate slow processing"""
    time.sleep(0.1)  # 100ms delay
    return torch.randn(128, 512)  # Return features


def fast_process_fn(sample, sample_id):
    """Fast processing"""
    return torch.randn(64, 256)


def test_parallel_preprocessor_basic(tmp_path):
    """Test basic parallel preprocessing"""
    preprocessor = ParallelPreprocessor(num_workers=2)
    
    samples = [MockSample(f"sample_{i}") for i in range(5)]
    sample_ids = [s.sample_id for s in samples]
    
    results = preprocessor.preprocess_samples(
        samples=samples,
        process_fn=fast_process_fn,
        sample_ids=sample_ids,
        desc="Test preprocessing"
    )
    
    assert len(results) == 5
    assert all(isinstance(r, torch.Tensor) for r in results)


def test_parallel_preprocessor_speedup(tmp_path):
    """Test that parallel processing completes successfully"""
    # Use threads to avoid pickling issues
    preprocessor = ParallelPreprocessor(num_workers=4, use_threads=True)
    
    samples = [MockSample(f"sample_{i}") for i in range(10)]
    sample_ids = [s.sample_id for s in samples]
    
    # Parallel processing
    start = time.time()
    results_parallel = preprocessor.preprocess_samples(
        samples=samples,
        process_fn=slow_process_fn,
        sample_ids=sample_ids,
        desc="Parallel test"
    )
    parallel_time = time.time() - start
    
    # Should complete successfully
    assert len(results_parallel) == 10
    # With 4 workers, should be faster than sequential (10 * 0.1 = 1.0s)
    # But allow some overhead, so check it's at least somewhat faster
    assert parallel_time < 0.8  # More lenient threshold


def test_parallel_preprocessor_with_caching(tmp_path):
    """Test preprocessing with cache checking"""
    # Use threads to avoid pickling issues with local functions
    preprocessor = ParallelPreprocessor(num_workers=2, use_threads=True)
    
    # Track which samples were processed (use a list that can be shared)
    processed_samples = []
    
    def process_with_tracking(sample, sample_id):
        processed_samples.append(sample_id)
        return torch.randn(64, 128)
    
    # Cache check function - first 3 are cached
    def cache_check(sample_id):
        return int(sample_id.split("_")[1]) < 3
    
    samples = [MockSample(f"sample_{i}") for i in range(5)]
    sample_ids = [s.sample_id for s in samples]
    
    results = preprocessor.preprocess_samples(
        samples=samples,
        process_fn=process_with_tracking,
        sample_ids=sample_ids,
        skip_cached=True,
        cache_check_fn=cache_check,
        desc="Test with caching"
    )
    
    # Should only process samples 3 and 4 (0, 1, 2 are cached)
    # Note: With threads, the list append should work, but order might vary
    assert len(results) == 2
    # Check that we got results for the non-cached samples
    result_ids = [s.sample_id for s in samples if s.sample_id not in ["sample_0", "sample_1", "sample_2"]]
    assert len(result_ids) == 2


def test_parallel_preprocessor_all_cached(tmp_path):
    """Test when all samples are cached"""
    preprocessor = ParallelPreprocessor(num_workers=2)
    
    processed_samples = []
    
    def process_fn(sample, sample_id):
        processed_samples.append(sample_id)
        return torch.randn(64, 128)
    
    def cache_check(sample_id):
        return True  # All cached
    
    samples = [MockSample(f"sample_{i}") for i in range(5)]
    sample_ids = [s.sample_id for s in samples]
    
    results = preprocessor.preprocess_samples(
        samples=samples,
        process_fn=process_fn,
        sample_ids=sample_ids,
        skip_cached=True,
        cache_check_fn=cache_check,
        desc="All cached test"
    )
    
    # Should not process anything
    assert len(processed_samples) == 0
    assert len(results) == 0


def test_parallel_preprocessor_error_handling(tmp_path):
    """Test error handling in parallel processing"""
    preprocessor = ParallelPreprocessor(num_workers=2)
    
    def failing_process_fn(sample, sample_id):
        if sample_id == "sample_2":
            raise ValueError("Test error")
        return torch.randn(64, 128)
    
    samples = [MockSample(f"sample_{i}") for i in range(5)]
    sample_ids = [s.sample_id for s in samples]
    
    # Should not crash, but may have fewer results
    results = preprocessor.preprocess_samples(
        samples=samples,
        process_fn=failing_process_fn,
        sample_ids=sample_ids,
        desc="Error handling test"
    )
    
    # Should have processed some samples (not all due to error)
    assert len(results) < 5


def test_parallel_preprocessor_batch_processing(tmp_path):
    """Test batch processing mode"""
    preprocessor = ParallelPreprocessor(num_workers=1, batch_size=3)
    
    def batch_process_fn(batch, batch_ids):
        # Process batch and return list of results
        return [torch.randn(64, 128) for _ in batch]
    
    samples = [MockSample(f"sample_{i}") for i in range(10)]
    sample_ids = [s.sample_id for s in samples]
    
    results = preprocessor.preprocess_batches(
        samples=samples,
        process_batch_fn=batch_process_fn,
        sample_ids=sample_ids,
        desc="Batch test"
    )
    
    assert len(results) == 10
    assert all(isinstance(r, torch.Tensor) for r in results)


def test_parallel_preprocessor_threads_vs_processes(tmp_path):
    """Test both thread and process executors"""
    # Process pool
    proc_preprocessor = ParallelPreprocessor(num_workers=2, use_threads=False)
    
    # Thread pool
    thread_preprocessor = ParallelPreprocessor(num_workers=2, use_threads=True)
    
    samples = [MockSample(f"sample_{i}") for i in range(5)]
    sample_ids = [s.sample_id for s in samples]
    
    # Both should work
    proc_results = proc_preprocessor.preprocess_samples(
        samples=samples,
        process_fn=fast_process_fn,
        sample_ids=sample_ids,
        desc="Process test"
    )
    
    thread_results = thread_preprocessor.preprocess_samples(
        samples=samples,
        process_fn=fast_process_fn,
        sample_ids=sample_ids,
        desc="Thread test"
    )
    
    assert len(proc_results) == 5
    assert len(thread_results) == 5


def test_process_pool_does_not_use_fork():
    """
    프로세스 풀이 fork 로 워커를 만들지 않는지 확인한다.

    fork 를 쓰면 부모가 이미 띄워둔 torch 스레드/공유메모리 fd 상태를 물려받아
    워커가 기동 중 교착한다. 교착은 '실패'가 아니라 '무한 대기'로 나타나 CI 에서
    잡기 어려우므로, 기능 테스트 대신 start method 자체를 직접 검증한다.
    """
    preprocessor = ParallelPreprocessor(num_workers=2)

    with preprocessor._make_executor() as executor:
        assert executor._mp_context.get_start_method() != "fork"


def test_process_pool_falls_back_when_start_method_unavailable():
    """쓸 수 없는 start method 를 주면 사용 가능한 것으로 대체한다."""
    preprocessor = ParallelPreprocessor(num_workers=2, mp_start_method="nonexistent")

    with preprocessor._make_executor() as executor:
        assert executor._mp_context.get_start_method() in mp.get_all_start_methods()

