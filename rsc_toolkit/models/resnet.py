from __future__ import annotations

from typing import Dict, Optional, Type

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .backbone.resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101


_RESNET_BACKBONES: Dict[str, Type[nn.Module]] = {
    "resnet10": ResNet10,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
}


@ModelFactory.register_model_class("resnet")
class ResNetClassifier(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int,
        backbone: str = "resnet18",
        pretrained: Optional[str] = None,
        track_bn: bool = True,
    ):
        super().__init__()
        backbone_key = backbone.lower()
        if backbone_key not in _RESNET_BACKBONES:
            raise ValueError(f"Unknown ResNet backbone: {backbone}. Available: {sorted(_RESNET_BACKBONES)}")

        self.backbone_name = backbone_key
        self.backbone = _RESNET_BACKBONES[backbone_key](track_bn=track_bn)
        self.classifier_head = nn.Linear(self.backbone.final_feat_dim, n_cls)

        if pretrained:
            pretrained_key = pretrained.lower()
            if pretrained_key == "sl":
                self.backbone.load_sl_official_weights()
            elif pretrained_key == "ssl":
                self.backbone.load_ssl_official_weights()
            else:
                raise ValueError("pretrained must be one of: null, 'sl', 'ssl'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected [B, F, T] or [B, 1, F, T], got {tuple(x.shape)}")
        feats = self.backbone(x)
        return self.classifier_head(feats)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):  # kwargs: base.py contract — kept open so it combines with data configs that attach meta
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
