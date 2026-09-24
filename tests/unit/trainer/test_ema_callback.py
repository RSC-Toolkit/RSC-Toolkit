# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import lightning as L

from rsc_toolkit.trainer.callbacks.ema import EMACallback


class _DummyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.rsc_model = nn.Linear(2, 1, bias=False)


def test_ema_updates_and_swaps_weights() -> None:
    pl_module = _DummyModule()
    cb = EMACallback(beta=0.5, model_attr="rsc_model", apply_to_validation=True, apply_to_test=False)

    trainer = SimpleNamespace(global_step=0)
    cb.on_train_start(trainer, pl_module)

    # step 0: ema initialized == current
    w0 = pl_module.rsc_model.weight.detach().clone()
    assert torch.allclose(cb.ema_params["weight"], w0)

    # step 1: change weight, run update => ema = 0.5*w0 + 0.5*w1
    with torch.no_grad():
        pl_module.rsc_model.weight.add_(1.0)
    w1 = pl_module.rsc_model.weight.detach().clone()
    trainer.global_step = 1
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

    expected = 0.5 * w0 + 0.5 * w1
    assert torch.allclose(cb.ema_params["weight"], expected)

    # validation swap + restore
    cb.on_validation_start(trainer, pl_module)
    assert torch.allclose(pl_module.rsc_model.weight.detach(), expected)
    cb.on_validation_end(trainer, pl_module)
    assert torch.allclose(pl_module.rsc_model.weight.detach(), w1)


def test_ema_state_roundtrip() -> None:
    pl_module = _DummyModule()
    cb = EMACallback(beta=0.9, model_attr="rsc_model", store_on_cpu=True)
    trainer = SimpleNamespace(global_step=0)
    cb.on_train_start(trainer, pl_module)

    state = cb.state_dict()
    cb2 = EMACallback(beta=0.5, model_attr="rsc_model")
    cb2.load_state_dict(state)

    assert cb2.ma.beta == 0.9
    assert cb2.store_on_cpu is True
    assert "weight" in cb2.ema_params
