import os
import pathlib
import hashlib
import json
import logging
from typing import Optional, Dict
import torch

logger = logging.getLogger(__name__)


class FeatureCache:
    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def configure(self, processor_config: Dict, processor_name: str):
        self.cache_key = self._create_cache_key(processor_config, processor_name)
        self.features_dir = self.cache_dir / "features" / self.cache_key
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def _create_cache_key(self, config: Dict, name: str) -> str:
        serialized = json.dumps(config, sort_keys=True, default=str)
        combined = f"{name}_{serialized}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _get_cache_path(self, sample_id: str) -> pathlib.Path:
        sample_hash = hashlib.md5(sample_id.encode()).hexdigest()
        return self.features_dir / f"{sample_hash}.pt"

    def is_cached(self, sample_id: str) -> bool:
        return self._get_cache_path(sample_id).exists()

    def get(self, sample_id: str) -> Optional[torch.Tensor]:
        path = self._get_cache_path(sample_id)
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Failed to load cache {path}: {e}")
            return None

    def put(self, sample_id: str, features: torch.Tensor):
        path = self._get_cache_path(sample_id)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

        # Write to a temp file first, then atomically replace.
        try:
            torch.save(features, tmp)
            os.replace(tmp, path)  # atomic on POSIX + Windows
        except Exception as e:
            logger.error(f"Failed to write cache for {sample_id}: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            
    def get_cache_stats(self) -> Dict[str, object]:
        """
        Computes stats by scanning the cache directory.
        Works safely with multiple DataLoader workers.
        """
        if not hasattr(self, "features_dir"):
            raise RuntimeError("Call configure(...) before get_cache_stats().")

        pt_files = list(self.features_dir.glob("*.pt"))
        total_files = len(pt_files)
        total_bytes = sum(p.stat().st_size for p in pt_files)

        # You may have temp files if a worker died mid-write; we can count them too.
        tmp_files = list(self.features_dir.glob("*.tmp.*"))
        tmp_bytes = sum(p.stat().st_size for p in tmp_files)

        return {
            "cache_dir": str(self.features_dir),
            "pt_files": total_files,
            "pt_total_bytes": total_bytes,
            "tmp_files": len(tmp_files),
            "tmp_total_bytes": tmp_bytes,
        }