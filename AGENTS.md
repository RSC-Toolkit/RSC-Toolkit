# Agent Instructions for RSC-Toolkit

## Scope and priorities
Focus on dataset downloading, processing, and dataloading. Treat training and
model execution as out of scope unless explicitly requested.

Priorities:
1) Keep data pipeline modular and testable.
2) Make it easy to add new datasets, processors, and augmentations.
3) Preserve reproducibility with readable configs and clear artifacts.

## Architecture guidelines
- Datasets:
  - Provide a consistent base interface for download/validate/process.
  - Keep dataset-specific logic in their own modules.
  - Support multiple sample levels (e.g., cycle and clip) using shared code.
- Processing/augmentations:
  - Allow processing at different stages (raw audio, feature, mel, etc.).
  - Make processors composable and explicit about inputs/outputs.
  - Make augmentation configs serializable for caching and reproduction.
- Caching:
  - Cache by dataset name + processor config + model/encoder settings + sample level.
  - Allow on-the-fly processing when no cache exists.
  - Provide a CLI path to precompute caches.
- Dataloaders:
  - Keep PyTorch Dataset/DataLoader wrappers thin.
  - Keep label extraction and sample indexing deterministic.
- Independence:
  - Avoid tight coupling to any trainer or model framework.
  - Prefer adapters so Hugging Face or PyTorch Lightning can plug in.

## Testing guidance
- Maintain tests for:
  - Download validation (mock network where possible).
  - Preprocessing + caching logic.
  - Dataloader output shapes, labels, and sample counts.
- Prefer small fixtures (toy dataset) for fast tests.
- Mark slow dataset tests and keep them opt-in.

## Code style
- Keep code self-documenting with clear names and structure.
- Add comments only for non-obvious logic or tradeoffs.
- Use type hints and small, focused modules.

## Config and logging
- Prefer readable config files that double as experiment documentation.
- Ensure logs include dataset name, sample level, and processor config.
- Avoid large config frameworks unless they add clear value.
