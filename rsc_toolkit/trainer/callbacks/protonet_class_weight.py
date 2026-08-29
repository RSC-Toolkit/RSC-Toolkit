"""
Prototype 논문 재현용: datamodule 이 계산한 balanced class weight 를 모델에 주입한다.

공식 `baseline_cnn/utils/data_generator.py:37-41` 의 `calculate_class_weight`
(sklearn compute_class_weight 'balanced', train 라벨 기준)를 그대로 쓴 값이며,
학습 손실 `F.nll_loss(..., weight=class_weight)` 에 들어간다(main_pytorch{,_prototype}.py).
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
            logger.warning("ProtoNetClassWeight: train_class_weight 또는 set_class_weight 없음 — 주입 생략")
            return
        model.set_class_weight(w)
        logger.info("ProtoNetClassWeight: 주입 %s", [round(float(x), 4) for x in w])
