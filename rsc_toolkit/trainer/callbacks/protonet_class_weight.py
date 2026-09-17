# Portions of this file are adapted from PrototypeSound (https://github.com/L3S/PrototypeSound); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
For the Prototype paper reproduction: injects the balanced class weight computed by the datamodule into the model.

The value follows `calculate_class_weight` in the official `baseline_cnn/utils/data_generator.py:37-41`
(sklearn compute_class_weight 'balanced', on train labels) and is used in
the training loss `F.nll_loss(..., weight=class_weight)` (main_pytorch{,_prototype}.py).
"""
from __future__ import annotations

import logging

import lightning as L

logger = logging.getLogger(__name__)


class ProtoNetClassWeight(L.Callback):
    def setup(self, trainer, pl_module, stage=None):
        dm = getattr(trainer, "datamodule", None)
        w = getattr(dm, "train_class_weight", None)
        model = getattr(pl_module, "rsc_model", pl_module)
        if w is None or not hasattr(model, "set_class_weight"):
            logger.warning("ProtoNetClassWeight: train_class_weight or set_class_weight missing; skipping injection")
            return
        model.set_class_weight(w)
        logger.info("ProtoNetClassWeight: injected %s", [round(float(x), 4) for x in w])
