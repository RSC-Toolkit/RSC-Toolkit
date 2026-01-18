from __future__ import annotations

from typing import Dict, Type, Optional

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .backbone.cnn import CNN6, CNN10, CNN14, load_cnn_checkpoint


_CNN_BACKBONES: Dict[str, Type[nn.Module]] = {
    "cnn6": CNN6,
    "cnn10": CNN10,
    "cnn14": CNN14,
}


class CNNClassifier(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int,
        backbone: str = "cnn10",
        dropout: float = 0.0,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        backbone_key = backbone.lower()
        if backbone_key not in _CNN_BACKBONES:
            raise ValueError(f"Unknown CNN backbone: {backbone}. Available: {sorted(_CNN_BACKBONES)}")
        self.backbone_name = backbone_key
        self.backbone = _CNN_BACKBONES[backbone_key](dropout=dropout)
        self.classifier_head = nn.Linear(self.backbone.final_feat_dim, n_cls)

        if pretrained_path:
            load_cnn_checkpoint(self.backbone, pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected [B, F, T] or [B, 1, F, T], got {tuple(x.shape)}")
        feats = self.backbone(x)
        return self.classifier_head(feats)

    def training_forward(self, audio_batch, labels, loss_func):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
