"""
Per-step CosineAnnealingLR, as in LoRA-RSC utils/engine_icbhi.py:34 (scheduler.step() after every optimizer.step),
main_icbhi.py:197 CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs) (eta_min 0, no warmup).

Closed form: lr(t) = eta_min + (lr0 - eta_min)·(1 + cos(π t / T_max)) / 2,  t = number of completed optimizer steps (first batch t=0 → lr0).
Implemented as a callback because PLAdapter's lr_scheduler is fixed to interval=epoch. The base lr is read from the optimizer param_groups (restored from the callback state on resume).
"""
from __future__ import annotations

import math
from typing import List, Optional

import lightning as L


class StepCosineLR(L.Callback):
    def __init__(self, epochs: int, eta_min: float = 0.0, steps_per_epoch: Optional[int] = None):
        super().__init__()
        self.epochs = int(epochs)
        self.eta_min = float(eta_min)
        self.steps_per_epoch = steps_per_epoch
        self.base_lrs: Optional[List[float]] = None
        self.t_max: Optional[int] = None

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "t_max": self.t_max}

    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict.get("base_lrs")
        self.t_max = state_dict.get("t_max")

    def on_train_start(self, trainer, pl_module):
        groups = [g for opt in trainer.optimizers for g in opt.param_groups]
        if self.base_lrs is None:
            self.base_lrs = [float(g["lr"]) for g in groups]
        if self.t_max is None:
            spe = self.steps_per_epoch or int(trainer.num_training_batches)
            self.t_max = max(int(spe) * self.epochs, 1)

    def lr_at(self, step: int, base: float) -> float:
        return self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * step / self.t_max)) / 2

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        t = int(trainer.global_step)
        i = 0
        for opt in trainer.optimizers:
            for g in opt.param_groups:
                g["lr"] = self.lr_at(t, self.base_lrs[i])
                i += 1
