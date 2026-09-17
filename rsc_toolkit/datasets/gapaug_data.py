"""
GaP-Aug (ICASSP 2024) training data -- class-balanced batch sampler (paper Sec. 3.1 'Instances are balanced sampled in each learning batch,
i.e., that same number of samples from each class are loaded in a batch'). **Re-implementation from the paper description** (no code released).

Assumptions (not confirmed by the authors):
- Batch 16 = 4 per class, sampled **with replacement** (uniformly) within each class, with a random shuffle inside the batch (so classes mix across the odd/even split).
- 'epoch' = steps_per_epoch (default 500) batches -> 20,000 iterations = 40 epochs. This makes Lightning epoch hooks (best-on-test evaluation, resume ckpt)
  run every 500 steps; the paper does not specify a 'validation interval' (it mentions no validation set).
- RNG: a new stream per epoch (seed = torch.initial_seed() + 7919*epoch). **Lightning 2.6 does not call set_epoch on the batch_sampler**
  -> the callback trainer/callbacks/gapaug_sampler_epoch.py calls set_epoch(current_epoch) at every on_train_epoch_start, so
  the epoch stream continues after resuming. `_epoch += 1` in __iter__ is the fallback when used without the callback.

val/test follow the parent IndexDataModule path unchanged (IndexDataset + WaveformProcessor, 10 s repeat padding).
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule, _tuple_collate

logger = logging.getLogger(__name__)


def collect_labels(ds) -> List[int]:
    """Collect only the label column from an IndexFeatureDataset / IndexDataset / ConcatDataset without reading audio."""
    if hasattr(ds, "index_ds"):
        return collect_labels(ds.index_ds)
    if isinstance(ds, ConcatDataset):
        out: List[int] = []
        for sub in ds.datasets:
            out.extend(collect_labels(sub))
        return out
    if hasattr(ds, "df"):
        return [int(v) for v in ds.df["label"].tolist()]
    return [int(ds[i][1]) for i in range(len(ds))]  # last resort (slow)


class BalancedBatchSampler(Sampler[List[int]]):
    """Draws per_class samples per class with replacement for every batch, shuffled within the batch. Length = steps_per_epoch."""

    def __init__(self, labels: Sequence[int], per_class: int = 4, steps_per_epoch: int = 500, n_classes: Optional[int] = None,
                 shuffle_within_batch: bool = True, seed: Optional[int] = None):
        labels_t = torch.as_tensor(list(labels), dtype=torch.long)
        self.n_classes = int(n_classes) if n_classes is not None else int(labels_t.max().item()) + 1
        self.per_class, self.steps_per_epoch = int(per_class), int(steps_per_epoch)
        self.shuffle_within_batch = bool(shuffle_within_batch)
        self.class_indices = [torch.nonzero(labels_t == c).flatten() for c in range(self.n_classes)]
        empty = [c for c, idx in enumerate(self.class_indices) if len(idx) == 0]
        if empty:
            raise ValueError(f"classes {empty} have no samples; cannot build balanced batches")
        self.seed = seed
        self._epoch = 0
        logger.info("BalancedBatchSampler: classes=%d per_class=%d steps/epoch=%d counts=%s",
                    self.n_classes, self.per_class, self.steps_per_epoch, [len(i) for i in self.class_indices])

    @property
    def batch_size(self) -> int:
        return self.per_class * self.n_classes

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        base = self.seed if self.seed is not None else int(torch.initial_seed()) % (2 ** 31)
        gen = torch.Generator().manual_seed((base + 7919 * self._epoch) % (2 ** 31))
        self._epoch += 1
        for _ in range(self.steps_per_epoch):
            picks = []
            for idx in self.class_indices:
                sel = torch.randint(len(idx), (self.per_class,), generator=gen)
                picks.append(idx[sel])
            batch = torch.cat(picks)
            if self.shuffle_within_batch:
                batch = batch[torch.randperm(len(batch), generator=gen)]
            yield batch.tolist()


class GaPAugDataModule(IndexDataModule):
    """Replaces only the train loader with BalancedBatchSampler (batch_size = per_class × n_classes; warns if it differs from config batch_size)."""

    def __init__(self, *args, gapaug: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(gapaug or {})
        self.per_class = int(cfg.get("per_class", 4))
        self.steps_per_epoch = int(cfg.get("steps_per_epoch", 500))
        self.n_classes = cfg.get("n_classes")
        self.sampler_seed = cfg.get("seed")
        self.train_sampler: Optional[BalancedBatchSampler] = None

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds is None. Did you forget to call setup('fit')?")
        labels = collect_labels(self.train_ds)
        self.train_sampler = BalancedBatchSampler(labels, self.per_class, self.steps_per_epoch, self.n_classes, seed=self.sampler_seed)
        if self.train_sampler.batch_size != self.batch_size:
            logger.warning("Balanced batch size %d (per_class %d × %d classes) differs from config batch_size %d -- using the sampler value",
                           self.train_sampler.batch_size, self.per_class, self.train_sampler.n_classes, self.batch_size)
        return DataLoader(
            self.train_ds, batch_sampler=self.train_sampler, num_workers=self.num_workers, collate_fn=_tuple_collate,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
