"""
**Two-stage training**, warm-up and class-weight injection of Domain Transfer (Wang & Wang, ICASSP 2022).

Stated in the paper (§3.2):
- "Adam optimizer with learning rate 6e-4, batch size 64, weight decay 1e-3, and **warm-up**"
- After training on augmented data, there is a separate stage that **fine-tunes again on the original data only with a small learning rate**.

Assumptions (not specified in the paper; all exposed as arguments):
- **Number of epochs not specified** → stage 1: 50 epochs, stage 2: 10 epochs.
- **Warm-up schedule not specified** → linear increase 0 → lr over the first 5 epochs of stage 1, then cosine decay.
- **Stage-2 lr not specified** → fixed at 0.1× the stage-1 lr (6e-5).

Turned off in stage 2 (= interpretation of "original data"):
- extra samples created by splicing are removed (`extra = []`)
- the three waveform augmentations (noise, time stretch, pitch shift) are disabled
- mixup is disabled (`model.mixup_alpha = 0`)
- circular padding is **not** turned off: the paper states it is also applied to the test set, so it is preprocessing, not augmentation.

The dataset length shrinks in stage 2, so `trainer.params.reload_dataloaders_every_n_epochs: 1` is required
(set in the same config).
"""
from __future__ import annotations

import logging
import math

import lightning as L

logger = logging.getLogger(__name__)


class DomainTransferStages(L.Callback):
    def __init__(
        self,
        learning_rate: float = 6.0e-4,
        stage1_epochs: int = 50,
        warmup_epochs: int = 5,
        stage2_lr_scale: float = 0.1,
        eta_min: float = 0.0,
    ):
        super().__init__()
        self.lr = float(learning_rate)
        self.stage1_epochs = int(stage1_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.stage2_lr = self.lr * float(stage2_lr_scale)
        self.eta_min = float(eta_min)
        self._switched = False

    # ------------------------------------------------------------------ weight injection
    def setup(self, trainer, pl_module, stage=None):
        dm = getattr(trainer, "datamodule", None)
        w = getattr(dm, "train_class_weight", None)
        model = getattr(pl_module, "rsc_model", pl_module)
        if w is None or not hasattr(model, "set_class_weight"):
            logger.warning("DomainTransferStages: skipping class weight injection (missing on datamodule or model)")
            return
        model.set_class_weight(w)
        logger.info("DomainTransferStages: injected class weight %s", [round(float(x), 4) for x in w])

    # ------------------------------------------------------------------ lr
    def lr_at(self, epoch: int) -> float:
        """epoch is 0-based. Stage 1: warm-up→cosine; stage 2: low fixed lr."""
        if epoch >= self.stage1_epochs:
            return self.stage2_lr
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.lr * float(epoch + 1) / float(self.warmup_epochs)
        span = max(1, self.stage1_epochs - self.warmup_epochs)
        prog = (epoch - self.warmup_epochs) / span
        return self.eta_min + (self.lr - self.eta_min) * (1 + math.cos(math.pi * prog)) / 2

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch == 0:
            # Same reason as PaperCosineLR: this callback overwrites lr every epoch, so a mismatch with the
            # optimizer group would silently nullify a CLI sweep (checked on fresh runs only).
            base_lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            if not math.isclose(base_lr, self.lr, rel_tol=1e-9):
                logger.warning(
                    "DomainTransferStages.learning_rate=%.3g differs from the optimizer lr=%.3g; "
                    "the effective lr is %.3g. Check the ${optimizer.params.lr} interpolation in the callbacks yaml.",
                    self.lr, base_lr, self.lr,
                )
        if epoch >= self.stage1_epochs and not self._switched:
            self._enter_stage2(trainer, pl_module)
        new_lr = self.lr_at(epoch)
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr

    # ------------------------------------------------------------------ stage-2 transition
    def _enter_stage2(self, trainer, pl_module) -> None:
        model = getattr(pl_module, "rsc_model", pl_module)
        if hasattr(model, "mixup_alpha"):
            model.mixup_alpha = 0.0
        dm = getattr(trainer, "datamodule", None)
        ds = getattr(dm, "train_ds", None)
        base = getattr(ds, "index_ds", ds)
        n_before = len(base) if base is not None else -1
        if base is not None:
            base.extra = []
            for flag in ("use_noise", "use_stretch", "use_pitch"):
                if hasattr(base, flag):
                    setattr(base, flag, False)
        self._switched = True
        logger.info(
            "Domain Transfer entering stage 2 (epoch %d): mixup, waveform augmentation and splice disabled, lr %.2e; training samples %d → %d",
            trainer.current_epoch, self.stage2_lr, n_before, len(base) if base is not None else -1,
        )
