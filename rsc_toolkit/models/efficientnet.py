from __future__ import annotations

from typing import Dict, Optional, Type

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .backbone.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2


_EFFICIENTNET_BACKBONES: Dict[str, Type[nn.Module]] = {
    "efficientnet_b0": EfficientNetB0,
    "efficientnet_b1": EfficientNetB1,
    "efficientnet_b2": EfficientNetB2,
}


@ModelFactory.register_model_class("efficientnet")
class EfficientNetClassifier(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int,
        backbone: str = "efficientnet_b0",
        pretrained: Optional[str] = None,
    ):
        super().__init__()
        backbone_key = backbone.lower()
        if backbone_key not in _EFFICIENTNET_BACKBONES:
            raise ValueError(
                f"Unknown EfficientNet backbone: {backbone}. Available: {sorted(_EFFICIENTNET_BACKBONES)}"
            )

        self.backbone_name = backbone_key
        self.backbone = _EFFICIENTNET_BACKBONES[backbone_key]()
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

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):  # kwargs: base.py 계약 — meta 를 붙이는 데이터 config 와도 조합되게 열어 둔다
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
