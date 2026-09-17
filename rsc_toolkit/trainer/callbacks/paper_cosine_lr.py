"""
Per-epoch cosine LR of the patch-mix family papers: reproduces util/misc.py adjust_learning_rate.

The original code has no scheduler object; it computes and overwrites lr at the start of every epoch:
  eta_min = lr * lr_decay_rate**3
  lr(e)   = eta_min + (lr - eta_min) * (1 + cos(pi * e / epochs)) / 2,  e = 1..epochs
Reproduced the same way (set directly in a callback). Warmup is not used by the AST scripts, so it is
not implemented (--warm unused).
"""
import logging
import math

import lightning as L

logger = logging.getLogger(__name__)


class PaperCosineLR(L.Callback):
    def __init__(self, learning_rate: float, epochs: int, lr_decay_rate: float = 0.1):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.lr_decay_rate = float(lr_decay_rate)

    def _check_single_source(self, trainer) -> None:
        """
        Warns if lr or epochs disagree with the optimizer/trainer groups.

        This callback overwrites lr every epoch, so on a mismatch an optimizer.params.lr
        sweep given on the CLI is silently ignored and results are attributed to the wrong
        hyperparameters (the callbacks yaml normally interpolates ${optimizer.params.lr}).
        Checked only on a fresh run (epoch 0); on resume the optimizer lr is a mid-schedule
        value and cannot be compared.
        """
        base_lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        if not math.isclose(base_lr, self.learning_rate, rel_tol=1e-9):
            logger.warning(
                "PaperCosineLR.learning_rate=%.3g differs from the optimizer lr=%.3g; "
                "this callback overwrites lr every epoch, so the effective lr is %.3g. "
                "Check whether the ${optimizer.params.lr} interpolation in the callbacks yaml is broken.",
                self.learning_rate, base_lr, self.learning_rate,
            )
        max_epochs = getattr(trainer, "max_epochs", None)  # may be absent on test stub trainers
        if max_epochs is not None and self.epochs != int(max_epochs):
            logger.warning(
                "PaperCosineLR.epochs=%d differs from trainer.max_epochs=%d; "
                "beyond epochs the cosine wraps around and lr rises again.",
                self.epochs, int(max_epochs),
            )

    def lr_at(self, epoch_1based: int) -> float:
        lr = self.learning_rate
        eta_min = lr * (self.lr_decay_rate ** 3)
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch_1based / self.epochs)) / 2

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self._check_single_source(trainer)
        new_lr = self.lr_at(trainer.current_epoch + 1)  # epochs start at 1 in the original code
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr
