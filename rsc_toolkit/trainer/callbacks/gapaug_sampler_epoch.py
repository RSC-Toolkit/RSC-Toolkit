"""
Epoch synchronization callback for the GaP-Aug balanced batch sampler.

Lightning 2.6 does not call set_epoch on a custom sampler passed as batch_sampler, so on resume (last.ckpt) the sampler would
replay the epoch-0 sequence. This callback calls datamodule.train_sampler.set_epoch(trainer.current_epoch) directly at every
on_train_epoch_start so the 'seed + 7919·epoch' sequence continues after resume (rsc_toolkit/datasets/gapaug_data.py BalancedBatchSampler).
"""
from __future__ import annotations

import logging

import lightning as L

logger = logging.getLogger(__name__)


class GaPAugSamplerEpoch(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        sampler = getattr(trainer.datamodule, "train_sampler", None)
        if sampler is None or not hasattr(sampler, "set_epoch"):
            logger.warning("GaPAugSamplerEpoch: datamodule has no train_sampler.set_epoch; skipping epoch synchronization")
            return
        sampler.set_epoch(int(trainer.current_epoch))
