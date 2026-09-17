"""
QLung (Audio-CLAP) — the **second backbone variant** in Tables 1 and 2 of the QLung paper.

Paper-reported values (arXiv 2606.11915v1, Sp/Se/Score):
  ICBHI official 60/40 : 81.98±3.82 / 44.81±3.47 / **63.39±0.40**   (+1.38 over the AST version, 62.01±1.18)
  SPRSound OOD         : 74.71±4.04 / 44.88±2.98 / **59.80±3.51**   (+1.57 over the AST version, 58.23±3.83)

**Grade C (re-implementation from the paper description).** The authors' repository (46408dd) contains only the AST path;
**there is no Audio-CLAP backbone code at all** (nor the SPRSound OOD evaluation code of Tables 1 and 2).
This file therefore combines **the paper description + the method part of the authors' AST code**, with the assumptions below.

Assumptions (3; not confirmed by the authors):
  ① **The method part is backbone-independent.** The AST version's head replacement (mlp_head → AngularClassifier s_a=37) and
     DFAM (s_d=15, κ=0.5, γ=0.5, m_target=0.2, λ=0.4) are applied unchanged on top of the CLAP features.
     Rationale: the paper presents both backbones as variants of the same method and gives no separate hyperparameters.
  ② **Features are CLAP audio_embeds (512-d)** — the same point as the Audio-CLAP baseline (models/clap.py).
     `qlung_dual_loss` applies F.normalize, so normalization is decided by the method part.
  ③ **AQS is computed from the waveform fed to the backbone.** The CLAP path runs at 48 kHz, so AQS values differ from
     the AST version (16 kHz) — the paper does not state the sampling rate used for AQS.

With `method=ce` the head is the same linear head as the Audio-CLAP baseline (no AngularClassifier), so the
(method − baseline) gain can be measured on the same data path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .backbone.clap import PretrainedCLAPWithProjection
from .factory import ModelFactory
from rsc_toolkit.methods.qlung import (
    ICBHI_TRAIN_CLASS_NUMS,
    AngularClassifier,
    AngularMarginLoss,
    qlung_dual_loss,
)


@ModelFactory.register_model_class("clap_qlung")
class QLungCLAP(BaseRSCModel, nn.Module):
    # prefixes excluded from MA (live weight averaging) — the original does not apply MA to DFAM either.
    ma_exclude_prefixes = ("dfam.",)
    accepts_meta = ("qlung_aqs",)  # batch[2] = AQS float (QLungDataModule)

    def __init__(
        self,
        n_cls: int = 4,
        method: str = "aqm",                 # "aqm" = QLung | "ce" = Audio-CLAP baseline (same data path)
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feat_dim: int = 512,
        s_a: float = 37.0,                   # AngularClassifier scale (train_qlung.py:271)
        s_d: float = 15.0,                   # DFAM scale (train_qlung.py:328)
        kappa: float = 0.5,                  # DFAM m (quality margin coefficient)
        gamma: float = 0.5,
        m_target: float = 0.2,
        lambda_aqm: float = 0.4,             # --alpha (DFAM CE weight)
        class_nums=None,                     # π_y — train class distribution of the dataset; ICBHI default if unset
        easy_margin: bool = False,
    ):
        super().__init__()
        if method not in {"aqm", "ce"}:
            raise ValueError(f"method must be aqm|ce, got {method!r}")
        self.method = method
        self.lambda_aqm = float(lambda_aqm)
        self.final_feat_dim = int(final_feat_dim)

        self.backbone = PretrainedCLAPWithProjection(
            pretrained_name=pretrained_name,
            final_feat_dim=final_feat_dim,
        )
        if method == "aqm":
            self.classifier = AngularClassifier(self.final_feat_dim, n_cls, s=s_a)
            self.dfam = AngularMarginLoss(
                self.final_feat_dim, n_cls, s=s_d, m=kappa, gamma=gamma, m_target=m_target,
                class_nums=list(class_nums) if class_nums is not None else list(ICBHI_TRAIN_CLASS_NUMS),
                easy_margin=easy_margin,
            )
        else:
            self.classifier = nn.Linear(self.final_feat_dim, n_cls)
            self.dfam = None

    # ------------------------------------------------------------------ features
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """CLAP audio_embeds (B, 512). No L2 normalization here — the method part applies it."""
        return self.backbone(x)

    def eval_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation path: normalized features → AngularClassifier (same convention as original validate:541-543)."""
        features = self.forward_features(x)
        if self.method == "ce":
            return self.classifier(features)
        return self.classifier(torch.nn.functional.normalize(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_logits(x)

    # ------------------------------------------------------------------ training
    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None, **kwargs):
        """
        meta_labels = AQS float (B,) — required for aqm training. loss_func = unweighted CE.
        Training and validation call the same method, so it branches on self.training (base.py contract).
        """
        if not self.training:
            logits = self.eval_logits(audio_batch)
            cast = logits.to(torch.float64) if self.method == "aqm" else logits
            return logits, loss_func(cast, labels)

        features = self.forward_features(audio_batch)

        if self.method == "ce":
            logits = self.classifier(features)
            return logits, loss_func(logits, labels)

        if meta_labels is None:
            raise ValueError(
                "clap_qlung(method=aqm) training requires an AQS float tensor as meta_labels "
                "(data=icbhi_qlung_clap)"
            )
        quality = meta_labels.to(features.device)
        if quality.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"AQS must be a float tensor: {quality.dtype}")
        logits, loss, _, _ = qlung_dual_loss(
            features, labels, quality, self.classifier, self.dfam, loss_func,
            lambda_aqm=self.lambda_aqm,
        )
        return logits, loss
