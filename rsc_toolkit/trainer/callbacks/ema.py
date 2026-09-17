# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn

from rsc_toolkit.trainer.utils.moving_average import MA


class EMACallback(L.Callback):
    """
    Maintains an exponential moving average (EMA) of model parameters and (optionally)
    swaps EMA weights in for validation/test.

    Configuration is intended for Hydra via `trainer.params.callbacks`.
    """

    def __init__(
        self,
        *,
        beta: float = 0.999,
        update_every_n_steps: int = 1,
        start_step: int = 0,
        model_attr: Optional[str] = "rsc_model",
        store_on_cpu: bool = False,
        apply_to_validation: bool = True,
        apply_to_test: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if update_every_n_steps < 1:
            raise ValueError(f"update_every_n_steps must be >= 1, got {update_every_n_steps}")
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")

        self.ma = MA(beta)
        self.update_every_n_steps = int(update_every_n_steps)
        self.start_step = int(start_step)
        self.model_attr = model_attr
        self.store_on_cpu = bool(store_on_cpu)
        self.apply_to_validation = bool(apply_to_validation)
        self.apply_to_test = bool(apply_to_test)

        self.ema_params: Dict[str, torch.Tensor] = {}
        self._backup_params: Optional[Dict[str, torch.Tensor]] = None

    def _resolve_model(self, pl_module: L.LightningModule) -> nn.Module:
        if self.model_attr:
            maybe = getattr(pl_module, self.model_attr, None)
            if isinstance(maybe, nn.Module):
                return maybe
        return pl_module

    def _iter_ema_parameters(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.is_floating_point(param.data):
                continue
            yield name, param

    def _as_ema_storage(self, t: torch.Tensor) -> torch.Tensor:
        if self.store_on_cpu:
            return t.detach().to(device="cpu", copy=True)
        return t.detach().clone()

    def _should_update(self, trainer: L.Trainer) -> bool:
        step = int(getattr(trainer, "global_step", 0) or 0)
        if step < self.start_step:
            return False
        return (step % self.update_every_n_steps) == 0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        model = self._resolve_model(pl_module)
        if self.ema_params:
            return
        for name, param in self._iter_ema_parameters(model):
            self.ema_params[name] = self._as_ema_storage(param.data)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._should_update(trainer):
            return

        model = self._resolve_model(pl_module)
        if not self.ema_params:
            for name, param in self._iter_ema_parameters(model):
                self.ema_params[name] = self._as_ema_storage(param.data)
            return

        for name, param in self._iter_ema_parameters(model):
            current = param.data.detach()
            if self.store_on_cpu:
                current = current.to(device="cpu", copy=True)
            self.ema_params[name] = self.ma.update_average(self.ema_params.get(name), current)

    def _apply_ema(self, pl_module: L.LightningModule) -> None:
        if self._backup_params is not None:
            return
        if not self.ema_params:
            return

        model = self._resolve_model(pl_module)
        backup: Dict[str, torch.Tensor] = {}
        for name, param in self._iter_ema_parameters(model):
            backup[name] = param.data.detach().clone()
            ema = self.ema_params.get(name)
            if ema is None:
                continue
            if ema.device != param.data.device:
                ema = ema.to(device=param.data.device)
            if ema.dtype != param.data.dtype:
                ema = ema.to(dtype=param.data.dtype)
            param.data.copy_(ema)
        self._backup_params = backup

    def _restore(self, pl_module: L.LightningModule) -> None:
        if self._backup_params is None:
            return
        model = self._resolve_model(pl_module)
        for name, param in self._iter_ema_parameters(model):
            old = self._backup_params.get(name)
            if old is not None:
                param.data.copy_(old)
        self._backup_params = None

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.apply_to_validation:
            self._apply_ema(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.apply_to_validation:
            self._restore(pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.apply_to_test:
            self._apply_ema(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.apply_to_test:
            self._restore(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.ma.beta,
            "update_every_n_steps": self.update_every_n_steps,
            "start_step": self.start_step,
            "model_attr": self.model_attr,
            "store_on_cpu": self.store_on_cpu,
            "apply_to_validation": self.apply_to_validation,
            "apply_to_test": self.apply_to_test,
            "ema_params": self.ema_params,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ma = MA(float(state_dict.get("beta", self.ma.beta)))
        self.update_every_n_steps = int(state_dict.get("update_every_n_steps", self.update_every_n_steps))
        self.start_step = int(state_dict.get("start_step", self.start_step))
        self.model_attr = state_dict.get("model_attr", self.model_attr)
        self.store_on_cpu = bool(state_dict.get("store_on_cpu", self.store_on_cpu))
        self.apply_to_validation = bool(state_dict.get("apply_to_validation", self.apply_to_validation))
        self.apply_to_test = bool(state_dict.get("apply_to_test", self.apply_to_test))
        self.ema_params = dict(state_dict.get("ema_params", {}))
        self._backup_params = None

