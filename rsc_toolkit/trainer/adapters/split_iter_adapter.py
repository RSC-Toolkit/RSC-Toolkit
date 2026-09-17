"""
split_iter(K) training adapter: reproduces the low-memory split-batch behavior of ce.py in the nttcslab/m2d app/icbhi_sprs patch.

train: whole batch through model.train_transform (batch min-max and SpecAug over B) → K forwards of D = B // K samples each
       → backward of the loss (sub-batch mean) accumulated → one optimizer.step. The remaining B - K·D samples are dropped (quirk of the original code; kept as part of the training recipe).
val:   model.val_transform on D = B // K samples at a time (min-max over D) → forward. The original ce.py leaves the remaining B - K·D samples out of evaluation
       (SPRSound test 1,429 = 64×22 + 21 → 1 sample dropped); the toolkit evaluates the remainder as a (K+1)-th sub-batch so that no test sample
       is dropped. The sub-batch composition and min-max of the first K·D samples match the original code, so their predictions are bit-identical; only the remainder is added.
The scheduler is stepped manually at epoch end (same position as scheduler.step() in the original code).
"""
from typing import Any, Dict, List, Optional

import torch

from .pl_adapter import PLAdapter


class SplitIterAdapter(PLAdapter):
    # M2D-family protocol outside the BaseRSCModel contract (training_forward). Checked at construction so that
    # a wrong combination with another model is not discovered mid-training as an AttributeError.
    REQUIRED_MODEL_ATTRS = ("train_transform", "val_transform")

    def __init__(self, *args, rsc_model=None, split_iter: int = 4, **kwargs):
        if rsc_model is None and args:
            rsc_model, args = args[0], args[1:]
        missing = [a for a in self.REQUIRED_MODEL_ATTRS if not callable(getattr(rsc_model, a, None))]
        if missing:
            raise TypeError(
                f"adapter=split_iter requires the model to have the methods {missing} (M2D-family protocol). "
                f"{type(rsc_model).__name__} only implements the BaseRSCModel contract (training_forward); "
                "use adapter=pl or implement these methods."
            )
        if int(split_iter) < 1:
            raise ValueError(f"split_iter must be >= 1: {split_iter!r}")
        super().__init__(*args, rsc_model=rsc_model, **kwargs)
        self.K = int(split_iter)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch[0].squeeze(1), batch[1]
        opt = self.optimizers()
        opt.zero_grad()
        x_t = self.rsc_model.train_transform(x)
        D = x_t.shape[0] // self.K
        total = 0.0
        logits_all, y_all = [], []
        for i in range(self.K):
            xs, ys = x_t[i * D:(i + 1) * D], y[i * D:(i + 1) * D]
            logits = self.rsc_model(xs)
            loss = self.loss_func(logits, ys)
            self.manual_backward(loss)
            total = total + loss.detach()
            logits_all.append(logits.detach()); y_all.append(ys)
        opt.step()
        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)
        for metric in self.train_metrics:
            metric.update(torch.cat(logits_all), torch.cat(y_all))
        return total

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def val_chunks(self, B: int) -> list:
        """Validation sub-batch bounds [(start, end), …]: K chunks of D=B//K plus one for the remainder (if any). If B < K, a single chunk. Every sample is evaluated exactly once."""
        D = B // self.K
        bounds = [(i * D, (i + 1) * D) for i in range(self.K)] if D > 0 else []
        if self.K * D < B:
            bounds.append((self.K * D, B))
        return bounds

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].squeeze(1), batch[1]
        losses, logits_all, y_all = [], [], []
        for s, e in self.val_chunks(int(x.shape[0])):
            xs, ys = x[s:e], y[s:e]
            logits = self.rsc_model(self.rsc_model.val_transform(xs))
            losses.append(self.loss_func(logits, ys))
            logits_all.append(logits); y_all.append(ys)
        self.log("val_loss", torch.stack(losses).mean(), prog_bar=True, on_step=False, on_epoch=True)
        for metric in self.val_metrics:
            metric.update(torch.cat(logits_all), torch.cat(y_all))
