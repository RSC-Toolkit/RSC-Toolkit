import logging
import multiprocessing as mp
from typing import List, Callable, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm


logger = logging.getLogger(__name__)


# 워커 프로세스를 fork 로 만들면, 부모가 이미 띄워둔 torch 의 스레드와 공유메모리
# 파일 디스크립터 상태(multiprocessing.resource_sharer)를 그대로 물려받아 워커가
# 기동 중 교착한다. process_fn 이 torch.Tensor 를 반환하면 결과를 돌려줄 때도 같은
# fd 전달 경로를 타므로 재현 확률이 높다.
# Linux 는 기본 start method 가 fork 라 이 문제가 드러나고, macOS 는 spawn 이라
# 드러나지 않는다. spawn 은 워커를 새 인터프리터로 띄워 그 상태를 물려받지 않으므로
# 두 플랫폼의 동작이 같아진다.
DEFAULT_MP_START_METHOD = "spawn"


class ParallelPreprocessor:
    """
    Handles parallel preprocessing of samples with progress tracking
    and resumable processing.
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        use_threads: bool = False,  # Use threads for GPU-bound tasks
        batch_size: Optional[int] = None,  # For batching GPU operations
        mp_start_method: Optional[str] = None,  # None이면 DEFAULT_MP_START_METHOD
    ):
        self.num_workers = num_workers
        self.use_threads = use_threads
        self.batch_size = batch_size
        self.mp_start_method = mp_start_method or DEFAULT_MP_START_METHOD

    def _make_executor(self):
        """스레드 풀 또는 (교착을 피하는 start method 를 쓰는) 프로세스 풀을 만든다."""
        if self.use_threads:
            return ThreadPoolExecutor(max_workers=self.num_workers)

        method = self.mp_start_method
        available = mp.get_all_start_methods()
        if method not in available:
            logger.warning(
                "multiprocessing start method '%s' 를 쓸 수 없어 '%s' 로 대체한다.",
                method,
                available[0],
            )
            method = available[0]

        return ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp.get_context(method),
        )
    
    def preprocess_samples(
        self,
        samples: List[Any],
        process_fn: Callable,
        sample_ids: Optional[List[str]] = None,
        desc: str = "Preprocessing",
        skip_cached: bool = True,
        cache_check_fn: Optional[Callable] = None
    ) -> List[Any]:
        """
        Preprocess samples in parallel.
        
        Args:
            samples: List of samples to process
            process_fn: Function that takes (sample, sample_id) and returns processed result
            sample_ids: Optional list of IDs for each sample
            desc: Progress bar description
            skip_cached: Skip samples that are already cached
            cache_check_fn: Function to check if sample is cached (sample_id -> bool)
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(samples))]
        
        # Filter out cached samples if requested
        if skip_cached and cache_check_fn:
            to_process = []
            to_process_ids = []
            for sample, sid in zip(samples, sample_ids):
                if not cache_check_fn(sid):
                    to_process.append(sample)
                    to_process_ids.append(sid)
            samples = to_process
            sample_ids = to_process_ids
            logger.info(f"Skipping {len(sample_ids) - len(to_process_ids)} cached samples")
        
        if not samples:
            logger.info("All samples already cached!")
            return []
        
        results = []
        with self._make_executor() as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(process_fn, sample, sid): sid
                for sample, sid in zip(samples, sample_ids)
            }
            
            # Process with progress bar
            iterator = tqdm(as_completed(future_to_id), total=len(future_to_id), desc=desc)
            
            for future in iterator:
                sample_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append((sample_id, result))
                except Exception as e:
                    logger.error(f"Error processing {sample_id}: {e}")
        
        # Sort results by original order
        id_to_result = dict(results)
        return [id_to_result.get(sid) for sid in sample_ids if sid in id_to_result]
    
    def preprocess_batches(
        self,
        samples: List[Any],
        process_batch_fn: Callable,
        sample_ids: Optional[List[str]] = None,
        desc: str = "Preprocessing batches"
    ) -> List[Any]:
        """
        Preprocess samples in batches (useful for GPU operations like CLAP).
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(samples))]
        
        if self.batch_size is None:
            self.batch_size = len(samples)
        
        results = []
        num_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        
        iterator = tqdm(range(0, len(samples), self.batch_size), desc=desc, total=num_batches)
        
        for i in iterator:
            batch = samples[i:i + self.batch_size]
            batch_ids = sample_ids[i:i + self.batch_size]
            
            try:
                batch_results = process_batch_fn(batch, batch_ids)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size}: {e}")
        
        return results

