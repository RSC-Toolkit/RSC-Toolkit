# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from typing import Callable, List, Optional, Any, Dict

import lightning as L
import torch.nn as nn
from torchmetrics import Metric

from ..factories.build_optimizer import build_optimizer


MetricFactory = Callable[[], Metric]


def check_meta_contract(rsc_model: Any, datamodule: Any) -> None:
    """
    Compatibility check of the batch[2] (meta) side channel, run once before training starts.

    Compared only when the datamodule declares `meta_kind` (what it sends) and the model declares
    `accepts_meta` (what it accepts). If either side does not declare it, the check passes (opt-in). On a mismatch it raises
    TypeError, blocking **combinations that silently train wrong**, such as interpreting an AQS float as a stethoscope id.
    """
    if datamodule is None:
        return
    kind = getattr(datamodule, "meta_kind", None)
    accepts = getattr(rsc_model, "accepts_meta", None)
    if kind is None or accepts is None:
        return
    accepts = (accepts,) if isinstance(accepts, str) else tuple(accepts)
    if kind not in accepts:
        raise TypeError(
            f"datamodule {type(datamodule).__name__} sends meta_kind={kind!r} as batch[2], but "
            f"model {type(rsc_model).__name__} does not list it in accepts_meta={accepts}: data/model combination error. "
            "(If the model does not use meta, remove the accepts_meta declaration or add that kind.)"
        )


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

    def setup(self, stage: Optional[str] = None) -> None:
        # If a datamodule is attached to the trainer, check the meta side-channel combination once.
        # (Skipped silently when there is no trainer or dataloaders are passed directly.)
        trainer = getattr(self, "_trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        check_meta_contract(self.rsc_model, datamodule)

    def training_step(self, batch, batch_idx):
        audio_batch = batch[0].squeeze(1)
        labels = batch[1]

        extra = {}
        if len(batch) > 2:
            # Present only when the datamodule's meta_label_mode is on (sgscl family: DAT, SG-SCL).
            # The receiving model's training_forward must support the meta_labels argument.
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
            # Models that also need meta for evaluation, such as BTS (bts_precomputed datamodule attaches it to val too).
            # The index datamodule does not attach meta to val batches, so the sgscl family never reaches here.
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
        # Training metrics get the `train/` prefix; validation keys (acc@1, icbhi_score, S_p, S_e) stay unprefixed.
        # Without the prefix, the same key would mix train and val in callback_metrics and one TensorBoard series, and putting
        # icbhi_score in the train list would let the best-checkpoint callback read the train value. Validation keys are unchanged,
        # so the best.json schema, verification scripts and existing runs stay compatible.
        for metric in self.train_metrics:
            self.log_dict({f"train/{k}": v for k, v in metric.compute().items()}, prog_bar=True)
            metric.reset()

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        cfg = dict(self.optimizer_cfg)
        sched_cfg = cfg.pop("lr_scheduler", None)
        optimizer = build_optimizer(cfg, self.parameters())
        if not sched_cfg:
            return optimizer
        # Per-epoch torch scheduler (e.g. SCL-metadata's CosineAnnealingLR(T_max=40, eta_min=1e-6);
        # the original code calls scheduler.step() at the end of every epoch, and Lightning steps at the same point with interval="epoch")
        import torch.optim.lr_scheduler as S

        name = sched_cfg["name"]
        params = dict(sched_cfg.get("params", {}))
        cls = {"cosine_annealing": S.CosineAnnealingLR, "step": S.StepLR, "multistep": S.MultiStepLR}[name]  # multistep: AMS-D main.py adjust_learning_rate
        # interval: default "epoch". Use "step" for papers whose original code calls scheduler.step() every iteration, such as Prototype (ICASSP 2022).
        interval = str(sched_cfg.get("interval", "epoch"))
        if interval not in {"epoch", "step"}:
            raise ValueError(f"lr_scheduler.interval must be epoch|step, got {interval!r}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": cls(optimizer, **params), "interval": interval}}
