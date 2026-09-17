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

    def configure(self, processor_config: Dict, processor_name: str, audio_chain: Optional[Dict] = None):
        """
        The cache key is built from the **processor config only** (n_mels, target_frames, normalization, ...).

        Features also depend on the **upstream audio chain** (resampling, segment cropping, padding, fade,
        time units), even with an identical processor config. The key does not capture that, so two configs
        with different chains sharing one `cache_dir` would **silently read each other's features** -- the kind
        of error that can invalidate a whole set of reproduction results.

        Putting the chain into the key would invalidate every existing cache and force hours of recomputation,
        so instead a **chain ledger (`_chain.json`) is written and any mismatch raises immediately**.
        - The ledger has the form `{dataset_name: chain}`. **Sample ids never overlap across datasets**, so
          different datasets can safely share one directory (ICBHI and SPRSound currently share features_cache).
        - Same dataset with a different chain stops the run and requires a separate `cache_dir`.
        """
        self.cache_key = self._create_cache_key(processor_config, processor_name)
        self.features_dir = self.cache_dir / "features" / self.cache_key
        self.features_dir.mkdir(parents=True, exist_ok=True)
        if audio_chain:
            self._check_audio_chain(dict(audio_chain))
        else:
            # audio_chain becomes a required argument in the next release; for now only warn --
            # without a chain ledger nothing prevents reusing one cache key with different preprocessing.
            logger.warning(
                "FeatureCache.configure(%s, %s) called without audio_chain -- without a chain ledger, features from "
                "different preprocessing can be read silently. Pass audio_chain (dict of dataset_name, sampling_rate, "
                "pad_type, fade, ...) at the call site (required from the next release).",
                processor_name, self.features_dir,
            )

    # ------------------------------------------------------------------ chain ledger
    def _check_audio_chain(self, chain: Dict) -> None:
        dataset = str(chain.pop("dataset_name", "?"))
        path = self.features_dir / "_chain.json"
        ledger: Dict[str, Dict] = {}
        if path.exists():
            try:
                ledger = json.loads(path.read_text())
            except Exception as e:      # a corrupted ledger only warns and is rewritten (the cache itself may be fine)
                logger.warning("Could not read the chain ledger (%s: %s) -- rewriting it", path, e)
                ledger = {}
        known = ledger.get(dataset)
        if known is not None and known != chain:
            raise RuntimeError(
                f"Feature cache audio chain mismatch -- the same cache key is being used with different preprocessing.\n"
                f"  cache: {self.features_dir}\n"
                f"  dataset: {dataset}\n"
                f"  existing: {known}\n"
                f"  requested: {chain}\n"
                f"Set a separate features.cache_dir in one of the two configs "
                f"(otherwise features built with different preprocessing are read silently)."
            )
        if known is None:
            ledger[dataset] = chain
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
            try:
                tmp.write_text(json.dumps(ledger, sort_keys=True, indent=2, default=str))
                os.replace(tmp, path)
            except Exception as e:
                logger.warning("Could not write the chain ledger (%s: %s)", path, e)

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
            
    def validate_entries(self, sample_ids) -> Dict[str, object]:
        """
        Check that the cache entries for the given sample_ids are intact.

        No feature shape is assumed -- shapes differ per model/feature type, e.g. CLAP embeddings (1, 512)
        vs. fbank (1024, 128). Only existence and loadability are checked.
        Callers that need shape checks inspect the returned entries themselves.
        """
        ids = [str(s) for s in sample_ids]
        missing = [sid for sid in ids if not self.is_cached(sid)]
        missing_set = set(missing)
        corrupted = [sid for sid in ids if sid not in missing_set and self.get(sid) is None]
        return {
            "total": len(ids),
            "missing": missing,
            "corrupted": corrupted,
            "ok": not missing and not corrupted,
        }

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