# Test Plan

Focus: dataset download, preprocessing, caching, and dataloaders.

## Test tiers
### Fast (default)
- No network calls.
- Use the toy dataset under `sample_data/toy_dataset`.
- Validate:
  - Dataset index building and label extraction.
  - Processor output shapes and dtypes.
  - Cache hit/miss behavior using temporary directories.
  - Dataloader returns consistent lengths and labels.

### Slow (opt-in)
- May read actual audio files and run full feature extraction.
- Mark with `@pytest.mark.slow`.
- Run with: `pytest -m slow`
- Skip with: `pytest -m "not slow"`

## Coverage targets
- Dataset download validation with mocked downloads.
- Preprocess + cache creation with processor signatures in the key.
- Dataloader behavior for both cycle and clip sample levels.

## Test data rules
- Keep fixtures minimal and deterministic.
- Avoid writing outside the test temp directory.
- Ensure cache paths are cleaned between tests.
