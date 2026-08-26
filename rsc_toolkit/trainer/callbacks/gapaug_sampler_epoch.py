"""
GaP-Aug 균형 배치 샘플러의 epoch 동기화 콜백.

Lightning 2.6 은 batch_sampler 로 넘긴 커스텀 샘플러에 set_epoch 를 호출하지 않는다(검토에서 확인) → 재개(last.ckpt) 시 샘플러가
epoch 0 의 열을 다시 재생한다. 이 콜백이 on_train_epoch_start 마다 datamodule.train_sampler.set_epoch(trainer.current_epoch) 를
직접 호출해 'seed + 7919·epoch' 열이 재개 후에도 이어지게 한다 (rsc_toolkit/datasets/gapaug_data.py BalancedBatchSampler).
"""
from __future__ import annotations

import logging

import lightning as L

logger = logging.getLogger(__name__)


class GaPAugSamplerEpoch(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        sampler = getattr(trainer.datamodule, "train_sampler", None)
        if sampler is None or not hasattr(sampler, "set_epoch"):
            logger.warning("GaPAugSamplerEpoch: datamodule 에 train_sampler.set_epoch 가 없다 — epoch 동기화 생략")
            return
        sampler.set_epoch(int(trainer.current_epoch))
