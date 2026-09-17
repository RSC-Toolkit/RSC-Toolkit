# Data Pipeline Proposal

This document proposes a clean, model-agnostic data pipeline focused on
download -> index -> load -> (optionally cache) -> return tensors.

The emphasis is on dataset download, preprocessing/indexing, feature processing,
and dataloaders. Training is intentionally out of scope.

## Goals
- Dataset handling independent of trainers and training frameworks.
- Processing that is configurable, composable, and cacheable.
- Support both cycle-level and clip-level sampling with shared logic.
- Clear extension points for new datasets, processors, and augmentations.
- Reproducible experiments via stable sample IDs and readable configs.

## Core concept: the sample index
The sample index is the contract between dataset-specific logic and model-agnostic
loading/processing. It is a deterministic table of samples and metadata:
- `sample_id` (stable ID)
- `recording_path`
- optional segment times: `t0`, `t1` (seconds or frames)
- `label` (or multi-label fields)
- split/group fields (patient/session/recording IDs)
- any dataset metadata needed for filtering, stats, or stratified splitting

Why it matters:
- Enables deterministic ordering and reproducibility.
- Avoids re-parsing annotations and re-building splits in every run.
- Allows fast dataset inspection (counts, durations, label balance) without
  touching feature extraction.

## Suggested interfaces

### Dataset
Responsibilities: download/validate raw files, build and persist the sample index.

Proposed base API:
```
class BaseDataset:
    name: str
    version: str | None

    def download(self) -> None: ...
    def validate_raw(self) -> None: ...
    def build_index(self) -> "SampleIndex": ...
```

Validation levels (recommended):
- Structural: expected files/dirs exist; non-empty; extract succeeded.
- Parse: annotations readable; rows reference existing recordings.
- Sanity: audio loadable; durations > 0; labels in allowed set.
- Optional (when versions/mirrors exist): manifest-based file sizes and/or hashes.

### Sample representation
Use a single schema so cycle vs clip is just metadata:
```
class Sample:
    sample_id: str
    recording_path: str
    t0: float | None
    t1: float | None
    label: int
    meta: dict
```

Clip-level samples have `t0=None, t1=None` (meaning “full recording”).

### Processor
Responsibilities: convert a `Sample` (raw audio slice) into model-ready features.

Proposed base API:
```
class BaseProcessor:
    name: str

    def signature(self) -> str: ...
    def process(self, sample: "Sample", audio: "Audio") -> "Feature": ...
```

Notes:
- `signature()` must include model/encoder settings (e.g., CLAP model name and
  preprocessing parameters). This is critical for correct caching.
- Processors can be composed (e.g., load -> resample -> mel -> normalize -> embed).

### Augmentation
Responsibilities: optional transforms at different stages.

Proposed base API:
```
class Augmentation:
    stage: Literal["raw", "feature", "mel"]
    deterministic: bool
    cacheable: bool
    def signature(self) -> str: ...
    def apply(self, x): ...
```

Caching guidance:
- Default: do not cache post-augmentation outputs (risk of cache explosion).
- Cache only deterministic stages by default; apply stochastic augmentation at runtime.
- Allow caching of augmentation outputs only when `deterministic and cacheable`,
  and only with an explicit opt-in policy.

### Cache
Responsibilities: store and load processed features for deterministic pipelines.

Cache key proposal (deterministic by default):
```
cache_key = hash(
    dataset.name,
    dataset.version,
    sample_level,
    processor.signature(),
)
```

If augmentations are cached (opt-in):
```
cache_key = hash(..., augmentation.signature())
```

Recommended cache controls:
- `cache.mode`: `off | read | readwrite`
- `cache.strict`: error on miss (useful after precompute)

### Dataloader
Responsibilities: provide `(features, label)` with optional caching.

Flow:
- Load the sample index.
- For each `sample_id`:
  - Load raw audio (and slice by `t0/t1` if present).
  - If `cache.mode in {read, readwrite}`: try cache read.
  - On miss: run processor; if `cache.mode == readwrite`, write feature.
  - Apply runtime-only augmentations (typically stochastic).
  - Return tensors + label(s).

## Optional: standardize raw audio
Some datasets are inconsistent or slow to decode. Consider an optional, separate
"audio normalization" step:
- Converts original audio into a local, standardized cache (e.g., mono, fixed SR).
- Keeps the sample index stable while improving load speed and consistency.
- This is distinct from the feature cache (mel/CLAP/etc).

## CLI hooks (data-focused)
- `datasets download <name>`: download and validate raw dataset.
- `datasets build-index <name>`: build and persist the sample index + stats.
- `datasets preprocess <name> --processor <id>`: iterate full index and populate feature cache.
- `datasets inspect <name>`: print dataset stats (counts, durations, label balance).

## Recommendations (defaults)
- Keep dataset processing deterministic and cheap: build an index + QA report.
- Cache deterministic feature pipelines by default; do not cache stochastic augmentation outputs.
- Make caching optional per run and per processor (`off/read/readwrite`).
- Prefer simple, readable configs for the data pipeline (YAML + pydantic/dataclasses).

## Open questions
- Cache format: `npz` vs `torch.save` vs `arrow` (pick one based on throughput needs).
- Index format: CSV/Parquet/JSONL (Parquet is great for large indices; CSV is simplest).
- Audio normalization cache format (if enabled): `wav` vs `flac`.
