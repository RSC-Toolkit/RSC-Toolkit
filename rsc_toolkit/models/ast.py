from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .backbone.ast import ASTBackbone


@ModelFactory.register_model_class("ast")
class ASTClassifier(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        pretrained: str = "imagenet",
        pretrained_path: str | None = None,
        download_if_missing: bool = True,
        model_size: str = "base384",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = ASTBackbone(
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            download_if_missing=download_if_missing,
            model_size=model_size,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.backbone.final_feat_dim),
            nn.Linear(self.backbone.final_feat_dim, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.dropout is not None:
            feats = self.dropout(feats)
        return self.classifier_head(feats)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):  # kwargs: base.py 계약 — meta 를 붙이는 데이터 config 와도 조합되게 열어 둔다
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
