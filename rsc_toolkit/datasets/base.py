from typing import ClassVar, Literal, Optional, List, Any, Type, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging
import functools
import pathlib
import hashlib
import json
import torch

import torchaudio

from rsc_toolkit.datasets.sample_index import REQUIRED_INDEX_COLUMNS as _REQUIRED_INDEX_COLUMNS


logger = logging.getLogger(__name__)


class BaseRSCDatasetDownloader(ABC, BaseModel):
    """
    Interface for downloading datasets
    """
    name: str
    outputs_directory: str
    force: bool = False

    @abstractmethod
    def validate_downloaded_files(self) -> bool:
        """
        Validate the downloaded files.
        """

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Download the dataset files
        """

    def download(self):
        outputs_path = pathlib.Path(self.outputs_directory).expanduser().resolve()
        logger.info("Dataset '%s' download directory: %s", self.name, outputs_path)

        if not self.force and self.validate_downloaded_files():
            logger.info(f"Found existing {self.name}, skipping download.")
            return
        logger.info("⬇️ Downloading %s", self.name)
        self.download_dataset()
        logger.info(f"✅ Completed {self.name} download")


class BaseRSCDatasetProcessor(ABC, BaseModel):
    """
    Interface for processing datasets for training
    """
    name: str # TODO better to use this from parent?
    outputs_directory: str
    force: bool = False
    preprocess_mode: Literal["full", "metadata_only", "none"] = "full"
    
    # Preprocessing options
    preprocess_features: bool = False
    feature_processor: Optional[Any] = None  # BaseAudioProcessor, but avoid circular import
    num_preprocess_workers: int = 4
    preprocess_batch_size: Optional[int] = None  # For GPU-based processors

    @functools.cached_property
    def cache_dir_path(self):

        def _create_unique_name(**kwargs):
            # Sort keys for determinism
            serialized = json.dumps(kwargs, sort_keys=True, default=str)
            # Hash for compact unique identifier
            return hashlib.md5(serialized.encode()).hexdigest()
        
        # Include feature processor config in cache key if present.
        #
        # 예전에는 `_get_processor_config()`(밑줄) 를 호출하고 AttributeError 를
        # 통째로 삼켰다. 실제 이름은 밑줄 없는 `get_processor_config()` 라서
        # 설정이 조용히 캐시 키에서 빠졌고, n_mels 가 달라도 같은 캐시를 쓰게 됐다.
        # 메서드 유무는 getattr 로 보고, 메서드 안에서 난 AttributeError 는
        # 삼키지 않고 그대로 드러낸다.
        kwargs = {}
        if self.feature_processor is not None:
            getter = getattr(self.feature_processor, "get_processor_config", None)
            if callable(getter):
                kwargs["feature_processor"] = getter()
            else:
                logger.warning(
                    "%s 에 get_processor_config() 가 없어 설정이 캐시 키에서 빠진다. "
                    "설정이 다른 피처가 같은 캐시를 공유할 수 있다.",
                    type(self.feature_processor).__name__,
                )

        key = _create_unique_name(
            preprocess_mode=self.preprocess_mode,
            **kwargs
        )

        cache_path = pathlib.Path(self.outputs_directory, "cache", key)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @functools.cached_property
    def index_dir_path(self):
        path = pathlib.Path(self.outputs_directory, "index")
        path.mkdir(parents=True, exist_ok=True)
        return path

    # 인덱스 검증에서 찾아볼 sample_level 들. 데이터셋이 일부만 만들면 줄여서 재정의한다.
    processed_sample_levels: Tuple[str, ...] = ("cycle", "clip")

    # 인덱스 CSV 가 반드시 갖춰야 하는 컬럼 (IndexDataset 이 요구하는 것과 같다).
    # Miika 가 추가한 `split` 컬럼은 옛 인덱스에 없을 수 있어 필수로 두지 않는다.
    # 단일 출처: rsc_toolkit.datasets.sample_index.REQUIRED_INDEX_COLUMNS (CSV_COLUMNS 에서 split 을 뺀 것)
    REQUIRED_INDEX_COLUMNS: ClassVar[Tuple[str, ...]] = _REQUIRED_INDEX_COLUMNS

    def processed_extra_files(self) -> List[str]:
        """
        index 디렉토리에 인덱스와 함께 있어야 하는 보조 산출물 이름 목록.
        데이터셋별로 재정의한다 (예: sprsound 의 official_split.txt).
        """
        return []

    def validate_index_labels(self, df: Any, *, sample_level: str) -> bool:
        """
        라벨 검증 훅. 라벨 공간은 데이터셋·과제마다 다르므로(ICBHI 4클래스, SPRSound
        이벤트 7종, clip 레벨은 라벨이 없기도 함) 기본은 최소한만 본다:
        cycle 레벨은 라벨 결측이 없어야 한다. 더 엄격한 규칙은 데이터셋이 재정의한다.
        """
        if sample_level != "cycle":
            return True
        missing = int(df["label"].isna().sum())
        if missing:
            logger.error("cycle 인덱스에 라벨 결측 %d/%d 행", missing, len(df))
            return False
        return True

    def _processed_index_dir(self) -> pathlib.Path:
        # index_dir_path 는 디렉토리를 만들어 버리므로(cached_property) 검증에서는 경로만 계산한다.
        return pathlib.Path(self.outputs_directory) / "index"

    def _validate_index_file(self, path: pathlib.Path, *, sample_level: str) -> bool:
        import pandas as pd

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.error("인덱스를 읽을 수 없다: %s (%s)", path, exc)
            return False

        missing_cols = [c for c in self.REQUIRED_INDEX_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.error("인덱스 컬럼 누락 %s: %s", missing_cols, path)
            return False

        if len(df) == 0:
            # build-index 가 exit 0 으로 끝나도 행이 0 이면 학습은 불가능하다 (실제로 겪은 결함).
            logger.error("인덱스에 행이 없다: %s", path)
            return False

        levels = set(df["sample_level"].astype(str).str.strip().str.lower())
        if levels != {sample_level}:
            logger.error("인덱스 %s 의 sample_level 값이 %s 다 (기대: %s)", path.name, sorted(levels), sample_level)
            return False

        stats_path = path.with_name(f"index_{sample_level}_stats.json")
        if stats_path.exists():
            try:
                total = json.loads(stats_path.read_text()).get("total_samples")
            except Exception as exc:
                logger.error("stats 파일을 읽을 수 없다: %s (%s)", stats_path, exc)
                return False
            if total is not None and int(total) != len(df):
                logger.error("stats 의 total_samples=%s 와 인덱스 행수 %d 가 다르다 (stale?): %s", total, len(df), stats_path)
                return False
        else:
            logger.warning("stats 파일이 없다 (선택 사항): %s", stats_path)

        if not self.validate_index_labels(df, sample_level=sample_level):
            return False

        logger.info("인덱스 OK: %s (%d 행)", path.name, len(df))
        return True

    def validate_processed_files(self) -> Optional[bool]:
        """
        전처리 산출물(인덱스와 보조 파일)이 온전한지 검사한다.

        반환:
          None  = 검사할 산출물이 아직 없다 (build-index 를 안 함). 실패가 아니다.
          False = 산출물이 있는데 깨져 있다 (컬럼 누락, 0행, stats 불일치, 보조 파일 누락 ...).
          True  = 존재하는 레벨의 인덱스가 모두 정상이고 보조 파일도 있다.

        존재하는 레벨만 검사한다 — cycle 만 쓰는 사용자가 clip 인덱스를 안 만들었다고
        실패로 치지 않기 위해서다. 피처 캐시는 모델/피처 종류마다 모양이 달라 여기서
        보지 않고 FeatureCache.validate_entries() 로 따로 본다.
        """
        index_dir = self._processed_index_dir()
        candidates = [(lvl, index_dir / f"index_{lvl}.csv") for lvl in self.processed_sample_levels]
        present = [(lvl, p) for lvl, p in candidates if p.exists()]
        if not present:
            logger.info("처리 산출물 없음 (아직 build-index 를 안 함): %s", index_dir)
            return None

        ok = True
        for lvl, path in present:
            ok = self._validate_index_file(path, sample_level=lvl) and ok

        for name in self.processed_extra_files():
            extra = index_dir / name
            if not extra.exists() or extra.stat().st_size == 0:
                logger.error("보조 산출물이 없거나 비어 있다: %s", extra)
                ok = False

        return ok

    @abstractmethod
    def process_dataset(self) -> None:
        """
        Process the dataset for running
        """

    def preprocess_all_features(self, samples: List[Any]) -> None:
        """
        Preprocess all samples with slow feature extractor.
        This runs in parallel and caches results.
        """
        if not self.feature_processor or not self.preprocess_features:
            return
        
        from rsc_toolkit.datasets.feature_cache import FeatureCache
        from rsc_toolkit.datasets.preprocessing import ParallelPreprocessor

        # 캐시가 아직 없으면 붙인다.
        #
        # 예전에는 여기서 processor 의 `cache_features` / `cache_dir` /
        # `_init_cache()` 를 건드렸는데, BaseAudioProcessor 에는 그런 속성이 없다.
        # 실제 API 는 `feature_cache` 하나다. 그래서 이 경로는 실제 processor 로
        # 부르면 AttributeError 로 죽었다.
        if self.feature_processor.feature_cache is None:
            cache = FeatureCache(cache_dir=self.cache_dir_path)
            cache.configure(
                processor_config=self.feature_processor.get_processor_config(),
                processor_name=type(self.feature_processor).__name__,
            )
            self.feature_processor.feature_cache = cache

        # Default to threads to avoid pickling issues with local functions
        # Threads work well for both CPU and GPU-bound tasks
        use_threads = True
        
        preprocessor = ParallelPreprocessor(
            num_workers=self.num_preprocess_workers,
            use_threads=use_threads,
            batch_size=self.preprocess_batch_size
        )
        
        def process_sample(sample, sample_id):
            # Extract audio from sample
            audio = self._extract_audio(sample)
            # Process with caching
            features = self.feature_processor.make_features_cached(
                audio, 
                sample_id=sample_id
            )
            return features
        
        def check_cached(sample_id):
            if self.feature_processor.feature_cache:
                return self.feature_processor.feature_cache.is_cached(sample_id)
            return False
        
        # Process all samples
        sample_ids = [self._get_sample_id(s) for s in samples]
        preprocessor.preprocess_samples(
            samples=samples,
            process_fn=process_sample,
            sample_ids=sample_ids,
            desc=f"Preprocessing {self.name} features",
            skip_cached=True,
            cache_check_fn=check_cached
        )
    
    def _extract_audio(self, sample: Any) -> torch.Tensor:
        """
        Extract audio tensor from sample.
        Override in subclasses if needed.
        """
        # Default implementation - subclasses should override
        if hasattr(sample, 'audio'):
            return sample.audio
        elif hasattr(sample, 'wav_path'):
            data, _ = torchaudio.load(str(sample.wav_path))
            return data
        else:
            raise NotImplementedError(f"Cannot extract audio from {type(sample)}")
    
    def _get_sample_id(self, sample: Any) -> str:
        """
        Get unique ID for sample (for caching).
        Override in subclasses.
        """
        if hasattr(sample, 'wav_path'):
            return str(sample.wav_path.stem)
        elif hasattr(sample, 'meta') and hasattr(sample.meta, 'patient_id'):
            return f"patient_{sample.meta.patient_id}"
        else:
            return str(hash(str(sample)))

    def process(self):
        if not self.force and self.validate_processed_files():
            logger.info(f"Found existing {self.name}, skipping processing.")
            return
        logger.info(f"⬇️ Processing {self.name}")
        self.process_dataset()
        logger.info(f"✅ Completed {self.name} processing")
    
class BaseRSCDataset(ABC, BaseModel):
    """
    Interface datasets
    """

    downloader_class: ClassVar[Type[BaseRSCDatasetDownloader]]
    processor_class: ClassVar[Type[BaseRSCDatasetProcessor]]

    downloader: BaseRSCDatasetDownloader
    processor: BaseRSCDatasetProcessor

    @classmethod
    def make_downloader(cls, **kwargs) -> BaseRSCDatasetDownloader:
        return cls.downloader_class(**kwargs)

    @classmethod
    def make_processor(cls, **kwargs) -> BaseRSCDatasetProcessor:
        return cls.processor_class(**kwargs)

    def download(self):
        self.downloader.download()

    def process(self):
        self.processor.process()
