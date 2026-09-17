# Portions of this file are adapted from RespireNet (https://github.com/microsoft/RespireNet); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
CNN model of RespireNet (EMBC 2021) — exact port of the official code.

Sources:
- Paper: RespireNet: A Deep Neural Network for Accurately Detecting Abnormal Lung Sounds in
  Limited Data Setting (https://arxiv.org/abs/2011.00196)
- Official code: https://github.com/microsoft/RespireNet nets/network_cnn.py (MIT)

Architecture (same as the original code): the fc of torchvision resnet34 (ImageNet) is replaced by
Dropout(0.5)-Linear(512,128)-ReLU-Dropout(0.5)-Linear(128,128)-ReLU + cls_fc Linear(128, n_cls).
Input is a 3-channel magma-colormap spectrogram image (B, 3, 192, 753) (processors/respirenet_image.py).
The attribute names model_ft / cls_fc are kept so the authors' released models/ckpt_best.pkl loads strictly.
fine_tune(block_layer): the original calls it only in eval.py (commented out in train.py) — not used for reproduction.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)


@ModelFactory.register_model_class("respirenet")
class RespireNetCNN(BaseRSCModel, nn.Module):
    accepts_meta = ("respirenet_chunk_index",)  # RespireNetDataModule attaches the chunk index (ignored by the model)

    def __init__(self, n_cls: int = 4, drop_prob: float = 0.5, imagenet_pretrain: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if imagenet_pretrain else None
        self.model_ft = torchvision.models.resnet34(weights=weights)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Sequential(
            nn.Dropout(drop_prob), nn.Linear(num_ftrs, 128), nn.ReLU(True),
            nn.Dropout(drop_prob), nn.Linear(128, 128), nn.ReLU(True),
        )
        self.cls_fc = nn.Linear(128, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_fc(self.model_ft(x))

    def fine_tune(self, block_layer: int = 5) -> None:
        """Original network_cnn.py:26-32 — freezes children 0..block_layer."""
        for idx, child in enumerate(self.model_ft.children()):
            if idx > block_layer:
                break
            for param in child.parameters():
                param.requires_grad = False

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)

    def load_reference_checkpoint(self, path) -> None:
        """Strictly loads the authors' ckpt_best.pkl (model_ft.* / cls_fc.* 222 keys)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
        logger.info("Loaded RespireNet authors' checkpoint: %s", path)
