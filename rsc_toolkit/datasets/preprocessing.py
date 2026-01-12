import logging
import multiprocessing as mp
from typing import List, Callable, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm


logger = logging.getLogger(__name__)



class ParallelPreprocessor:
    """
    Handles parallel preprocessing of samples with progress tracking
    and resumable processing.
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        use_threads: bool = False,  # Use threads for GPU-bound tasks
        batch_size: Optional[int] = None  # For batching GPU operations
    ):
        self.num_workers = num_workers
        self.use_threads = use_threads
        self.batch_size = batch_size
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
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
        with self.executor_class(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(process_fn, sample, sid): sid
                for sample, sid in zip(samples, sample_ids)
            }
            
            # Process with progress bar
            if HAS_TQDM:
                iterator = tqdm(as_completed(future_to_id), total=len(future_to_id), desc=desc)
            else:
                iterator = as_completed(future_to_id)
            
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
        
        if HAS_TQDM:
            iterator = tqdm(range(0, len(samples), self.batch_size), desc=desc, total=num_batches)
        else:
            iterator = range(0, len(samples), self.batch_size)
        
        for i in iterator:
            batch = samples[i:i + self.batch_size]
            batch_ids = sample_ids[i:i + self.batch_size]
            
            try:
                batch_results = process_batch_fn(batch, batch_ids)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size}: {e}")
        
        return results

