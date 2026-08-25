from typing import Callable, List, Optional, Any, Dict

import lightning as L
import torch.nn as nn
from torchmetrics import Metric

from ..factories.build_optimizer import build_optimizer


MetricFactory = Callable[[], Metric]


class PLAdapter(L.LightningModule):
    def __init__(
        self,
        rsc_model: nn.Module,
        loss_func: nn.Module,
        optimizer_cfg: Dict[str, Any],
        train_metric_fns: Optional[List[MetricFactory]] = None,
        val_metric_fns: Optional[List[MetricFactory]] = None,
    ):
        super().__init__()
        self.rsc_model = rsc_model
        self.loss_func = loss_func
        self.optimizer_cfg = optimizer_cfg

        # Must instantiate inside to avoid shared state and ensure ownership
        self.train_metrics = nn.ModuleList([fn() for fn in (train_metric_fns or [])])
        self.val_metrics = nn.ModuleList([fn() for fn in (val_metric_fns or [])])

    def training_step(self, batch, batch_idx):
        audio_batch = batch[0].squeeze(1)
        labels = batch[1]

        extra = {}
        if len(batch) > 2:
            # datamodule 의 meta_label_mode 가 켜진 경우에만 존재 (sgscl 계열 DAT·SG-SCL).
            # 받는 모델의 training_forward 는 meta_labels 인자를 지원해야 한다.
            extra["meta_labels"] = batch[2]

        logits, loss = self.rsc_model.training_forward(
            audio_batch=audio_batch,
            labels=labels,
            loss_func=self.loss_func,
            **extra,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        for metric in self.train_metrics:
            metric.update(logits, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_batch = batch[0].squeeze(1)
        labels = batch[1]

        extra = {}
        if len(batch) > 2:
            # BTS 처럼 평가에도 meta 가 필요한 모델 (bts_precomputed datamodule 은 val 에도 붙인다).
            # index datamodule 은 val 배치에 meta 를 붙이지 않으므로 sgscl 계열은 여기 안 온다.
            extra["meta_labels"] = batch[2]

        logits, loss = self.rsc_model.training_forward(
            audio_batch=audio_batch,
            labels=labels,
            loss_func=self.loss_func,
            **extra,
        )

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        for metric in self.val_metrics:
            metric.update(logits, labels)

    def on_train_start(self):
        self.rsc_model.train()

    def on_train_epoch_end(self):
        for metric in self.train_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        return build_optimizer(self.optimizer_cfg, self.parameters())
