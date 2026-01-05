import pathlib
import hashlib
import json
import logging
from typing import Optional, Any, Dict
import torch

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Manages caching of expensive preprocessed features.
    Uses content-based hashing to invalidate cache when inputs change.
    """
    
    def __init__(
        self,
        cache_dir: pathlib.Path,
        processor_config: Dict[str, Any],
        processor_name: str
    ):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique cache key from processor config
        self.cache_key = self._create_cache_key(processor_config, processor_name)
        self.features_dir = self.cache_dir / "features" / self.cache_key
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track what's cached
        self.metadata_file = self.features_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _create_cache_key(self, config: Dict, name: str) -> str:
        """Create deterministic cache key from config"""
        # Sort for determinism
        serialized = json.dumps(config, sort_keys=True, default=str)
        combined = f"{name}_{serialized}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}, starting fresh")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_cache_path(self, sample_id: str) -> pathlib.Path:
        """Get cache file path for a sample"""
        # Use hash of sample_id to avoid filesystem issues with long names
        sample_hash = hashlib.md5(sample_id.encode()).hexdigest()
        return self.features_dir / f"{sample_hash}.pt"
    
    def get(self, sample_id: str) -> Optional[torch.Tensor]:
        """Retrieve cached features for a sample"""
        if sample_id not in self.metadata:
            return None
        
        cache_path = self._get_cache_path(sample_id)
        if not cache_path.exists():
            logger.warning(f"Metadata says {sample_id} is cached, but file missing")
            return None
        
        try:
            return torch.load(cache_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Error loading cache for {sample_id}: {e}")
            return None
    
    def put(self, sample_id: str, features: torch.Tensor):
        """Cache features for a sample"""
        cache_path = self._get_cache_path(sample_id)
        try:
            torch.save(features, cache_path)
            self.metadata[sample_id] = {
                "cached": True,
                "shape": list(features.shape),
                "dtype": str(features.dtype)
            }
            self._save_metadata()
        except Exception as e:
            logger.error(f"Error caching {sample_id}: {e}")
    
    def is_cached(self, sample_id: str) -> bool:
        """Check if sample is cached"""
        return sample_id in self.metadata and self._get_cache_path(sample_id).exists()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache"""
        total = len(self.metadata)
        existing = sum(1 for sid in self.metadata if self._get_cache_path(sid).exists())
        return {
            "total_entries": total,
            "existing_files": existing,
            "cache_dir": str(self.features_dir)
        }

