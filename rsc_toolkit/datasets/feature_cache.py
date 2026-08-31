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
        캐시 키는 **프로세서 설정만** 으로 만든다(n_mels·target_frames·정규화 등).

        그런데 같은 프로세서 설정이라도 **그 앞단의 오디오 체인**(리샘플·구간 절단·패딩·fade·시간 단위)이 다르면
        피처가 달라진다. 키에는 그 정보가 없으므로, 체인이 다른 두 config 가 같은 `cache_dir` 를 쓰면
        **조용히 남의 피처를 읽는다**. 실제로 재현 결과를 통째로 무효화할 수 있는 종류의 사고다.

        키 자체에 체인을 넣으면 기존 캐시가 전부 무효가 되어 수 시간을 다시 계산해야 하므로,
        대신 **체인 대장(`_chain.json`)을 남기고 어긋나면 즉시 예외**를 던진다.
        - 대장은 `{dataset_name: chain}` 형태다. **데이터셋이 다르면 sample_id 가 겹치지 않으므로**
          같은 디렉터리를 공유해도 안전하다(ICBHI 와 SPRSound 가 features_cache 를 함께 쓰는 현재 구조).
        - 같은 데이터셋인데 체인이 다르면 실행을 멈추고 `cache_dir` 분리를 요구한다.
        """
        self.cache_key = self._create_cache_key(processor_config, processor_name)
        self.features_dir = self.cache_dir / "features" / self.cache_key
        self.features_dir.mkdir(parents=True, exist_ok=True)
        if audio_chain:
            self._check_audio_chain(dict(audio_chain))

    # ------------------------------------------------------------------ 체인 대장
    def _check_audio_chain(self, chain: Dict) -> None:
        dataset = str(chain.pop("dataset_name", "?"))
        path = self.features_dir / "_chain.json"
        ledger: Dict[str, Dict] = {}
        if path.exists():
            try:
                ledger = json.loads(path.read_text())
            except Exception as e:      # 손상된 대장은 경고만 하고 새로 쓴다(캐시 자체는 멀쩡할 수 있다)
                logger.warning("체인 대장을 읽지 못했다 (%s: %s) — 새로 쓴다", path, e)
                ledger = {}
        known = ledger.get(dataset)
        if known is not None and known != chain:
            raise RuntimeError(
                f"피처 캐시 오디오 체인 불일치 — 같은 캐시 키를 서로 다른 전처리로 쓰려 한다.\n"
                f"  캐시: {self.features_dir}\n"
                f"  데이터셋: {dataset}\n"
                f"  기존: {known}\n"
                f"  요청: {chain}\n"
                f"둘 중 하나의 config 에서 features.cache_dir 를 분리하라 "
                f"(그대로 두면 다른 전처리로 만든 피처를 조용히 읽게 된다)."
            )
        if known is None:
            ledger[dataset] = chain
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
            try:
                tmp.write_text(json.dumps(ledger, sort_keys=True, indent=2, default=str))
                os.replace(tmp, path)
            except Exception as e:
                logger.warning("체인 대장을 쓰지 못했다 (%s: %s)", path, e)

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
        주어진 sample_id 들의 캐시가 온전한지 본다.

        피처의 모양은 가정하지 않는다 — CLAP 임베딩(1, 512)과 fbank(1024, 128)처럼
        모델/피처 종류마다 다르기 때문이다. 존재 여부와 로드 가능 여부만 본다.
        모양까지 맞추고 싶으면 호출자가 반환된 항목을 직접 검사한다.
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