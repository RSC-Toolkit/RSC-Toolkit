# Portions of this file are ported from PAFA (https://github.com/wa976/PAFA), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
BEATs fine-tuning model from the PAFA paper (Interspeech 2025) — exact port of the official code.

Sources:
- Paper: Patient-Aware Feature Alignment for Robust Lung Sound Classification:
  Cohesion-Separation and Global Alignment Losses (https://arxiv.org/abs/2505.23834)
- Official code: https://github.com/wa976/PAFA models/beats.py (BEATsTransferLearningModel)
  + the beats branch of main.py set_model()/train()/validate().

Architecture (same as the original code):
- Backbone: BEATs (built from the cfg of the iter3+ AS2M pretrained ckpt, no predictor). Input is a
  16 kHz waveform (B, L). extract_features internally computes a kaldi fbank (128 mel, 25/10 ms) →
  patch embedding → Transformer and returns a token sequence (B, T', 768).
- Classifier: nn.Linear(768, n_cls) applied **per token**, then averaged over time (main.py:365-367).
- PAFA: ProjectionHead (768→768, attention pooling, LayerNorm) produces (B, 768) and
  PAFALoss(patient_ids) is added: loss = w_ce*CE + w_pafa*PAFA (main.py:379-389).
- No SpecAugment, since the script uses --nospec (spec_transform=None).

Toolkit wiring: the datamodule yields (waveform, label, patient ID) batches (meta_label_mode=patient).
"""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .beats_upstream import BEATs, BEATsConfig
from .factory import ModelFactory

logger = logging.getLogger(__name__)

BEATS_CKPT_CANDIDATES = (
    "pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt",
    "pretrained_models/BEATs_iter3_plus_AS2M.pt",  # location used by the original code
)


def _resolve_beats_ckpt(path: Optional[str]) -> pathlib.Path:
    from rsc_toolkit.paths import resolve_pretrained

    if path is not None:
        return resolve_pretrained(explicit=path)
    found = resolve_pretrained(BEATS_CKPT_CANDIDATES)  # RSCTK_PRETRAINED_ROOT, then cwd
    if found is not None:
        return found
    raise FileNotFoundError(
        "BEATs_iter3_plus_AS2M.pt not found. Download 'BEATs_iter3+ (AS2M)' from microsoft/unilm/beats and "
        f"place it at {BEATS_CKPT_CANDIDATES[0]}"
    )


class BEATsTransferLearningModel(nn.Module):
    """Same as the original models/beats.py (builds BEATs from the ckpt cfg, loads weights, returns extract_features)."""

    def __init__(self, num_target_classes: int = 4, model_path: Optional[str] = None, spec_transform=None):
        super().__init__()
        self.num_target_classes = num_target_classes
        ckpt_path = _resolve_beats_ckpt(model_path)
        self.checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = BEATsConfig({
            **self.checkpoint["cfg"],
            "predictor_class": self.num_target_classes,
            "finetuned_model": False,
            "spec_transform": spec_transform,
        })
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])
        self.final_feat_dim = self.cfg.encoder_embed_dim
        del self.checkpoint  # the original keeps it, but it is not part of state_dict, so free the memory

    def forward(self, x, padding_mask=None, training=False):
        if padding_mask is not None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)
        return x


@ModelFactory.register_model_class("beats")
class BEATsClassifier(BaseRSCModel, nn.Module):
    """beats configuration of the original set_model: backbone + per-token Linear (mean pooling) [+ PAFA ProjectionHead]."""

    accepts_meta = ("patient",)  # batch[2] = patient ID (meta_label_mode=patient; required for method=pafa)

    def __init__(
        self,
        n_cls: int = 4,
        pretrained_path: Optional[str] = None,
        method: str = "ce",             # "ce" | "pafa"
        # ---- PAFA (defaults follow the beats_pafa.sh script) ----
        w_ce: float = 1.0,
        w_pafa: float = 1.0,
        lambda_pcsl: float = 50.0,
        lambda_gpal: float = 0.0005,
        hidden_dim: Optional[int] = None,
        output_dim: int = 768,
        norm_type: str = "ln",
        proj_type: str = "end2end",
    ):
        super().__init__()
        if method not in {"ce", "pafa"}:
            raise ValueError(f"method must be ce|pafa, got {method!r}")
        self.backbone = BEATsTransferLearningModel(num_target_classes=n_cls, model_path=pretrained_path)
        self.classifier = nn.Linear(self.backbone.final_feat_dim, n_cls)
        self.method = method
        self.w_ce, self.w_pafa = w_ce, w_pafa
        self.lambda_pcsl, self.lambda_gpal = lambda_pcsl, lambda_gpal
        if method == "pafa":
            from rsc_toolkit.methods.pafa import PAFALoss, ProjectionHead

            # original main.py:250 — beats uses attention=True
            self.projector = ProjectionHead(
                self.backbone.final_feat_dim, hidden_dim, output_dim,
                attention=True, norm_type=norm_type, proj_type=proj_type,
            )
            self.pafa_loss = PAFALoss()
        else:
            self.projector = nn.Identity()
            self.pafa_loss = None

    def forward_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (B, 1, L) -> (B, L)
        return self.backbone(waveform)      # (B, T', 768)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # original: output = classifier(features); output = output.mean(dim=1)
        return self.classifier(self.forward_features(waveform)).mean(dim=1)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        features = self.forward_features(audio_batch)
        logits = self.classifier(features).mean(dim=1)
        loss_class = loss_func(logits, labels)
        if (not self.training) or self.method == "ce":
            return logits, loss_class
        if meta_labels is None:
            raise ValueError("method=pafa requires patient IDs in the batch (datamodule meta_label_mode=patient)")
        proj = self.projector(features)
        loss_pafa = self.pafa_loss(proj, meta_labels, lambda_pcsl=self.lambda_pcsl, lambda_gpal=self.lambda_gpal)
        return logits, self.w_ce * loss_class + self.w_pafa * loss_pafa
