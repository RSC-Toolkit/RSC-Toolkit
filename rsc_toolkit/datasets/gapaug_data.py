"""
GaP-Aug(ICASSP 2024) 학습 데이터 — 클래스 균형 배치 샘플러 (논문 §3.1 'Instances are balanced sampled in each learning batch,
i.e., that same number of samples from each class are loaded in a batch'). **논문 명세 재구현** (코드 없음).

가정(저자 확인 전 — AUTHOR_QUESTIONS GaP-Aug #8):
- 배치 16 = 클래스별 4개, 클래스 안에서는 **복원 추출**(균등), 배치 안 순서는 무작위 셔플(홀/짝 분할에 클래스가 섞이도록).
- 'epoch' = steps_per_epoch(기본 500) 배치 → 20,000 iterations = 40 epoch. Lightning epoch 훅(best-on-test 평가·재개 ckpt)이
  500 step 마다 돌게 하는 장치이며 논문의 '검증 주기' 는 미기재(논문은 검증셋 언급 없음).
- 난수: epoch 마다 새 열 (seed = torch.initial_seed() + 7919·epoch). **Lightning 2.6 은 batch_sampler 의 set_epoch 를 호출하지 않는다**
  (검토에서 확인) → 콜백 trainer/callbacks/gapaug_sampler_epoch.py 가 on_train_epoch_start 마다 set_epoch(current_epoch) 를 호출해
  재개 후에도 epoch 열이 이어진다. __iter__ 의 _epoch += 1 은 콜백 없이 쓸 때의 fallback.

val/test 는 부모 IndexDataModule 경로 그대로 (IndexDataset + WaveformProcessor, 10 s repeat 패딩).
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule, _tuple_collate

logger = logging.getLogger(__name__)


def collect_labels(ds) -> List[int]:
    """IndexFeatureDataset / IndexDataset / ConcatDataset 에서 오디오를 읽지 않고 라벨 열만 모은다."""
    if hasattr(ds, "index_ds"):
        return collect_labels(ds.index_ds)
    if isinstance(ds, ConcatDataset):
        out: List[int] = []
        for sub in ds.datasets:
            out.extend(collect_labels(sub))
        return out
    if hasattr(ds, "df"):
        return [int(v) for v in ds.df["label"].tolist()]
    return [int(ds[i][1]) for i in range(len(ds))]  # 최후 수단(느림)


class BalancedBatchSampler(Sampler[List[int]]):
    """배치마다 클래스별 per_class 개를 복원 추출, 배치 내 셔플. 길이 = steps_per_epoch."""

    def __init__(self, labels: Sequence[int], per_class: int = 4, steps_per_epoch: int = 500, n_classes: Optional[int] = None,
                 shuffle_within_batch: bool = True, seed: Optional[int] = None):
        labels_t = torch.as_tensor(list(labels), dtype=torch.long)
        self.n_classes = int(n_classes) if n_classes is not None else int(labels_t.max().item()) + 1
        self.per_class, self.steps_per_epoch = int(per_class), int(steps_per_epoch)
        self.shuffle_within_batch = bool(shuffle_within_batch)
        self.class_indices = [torch.nonzero(labels_t == c).flatten() for c in range(self.n_classes)]
        empty = [c for c, idx in enumerate(self.class_indices) if len(idx) == 0]
        if empty:
            raise ValueError(f"클래스 {empty} 에 샘플이 없어 균형 배치를 만들 수 없다")
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
    """train 로더만 BalancedBatchSampler 로 교체 (batch_size = per_class × n_classes; config batch_size 와 다르면 경고)."""

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
            logger.warning("균형 배치 크기 %d (per_class %d × %d 클래스) 가 config batch_size %d 와 다르다 — 샘플러 값을 쓴다",
                           self.train_sampler.batch_size, self.per_class, self.train_sampler.n_classes, self.batch_size)
        return DataLoader(
            self.train_ds, batch_sampler=self.train_sampler, num_workers=self.num_workers, collate_fn=_tuple_collate,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
