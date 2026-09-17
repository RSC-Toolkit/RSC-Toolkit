# Portions of this file are ported from PC-MCL (https://github.com/wa976/PC-MCL), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
BEATs model of PC-MCL (arXiv 2601.17080; wa976/PC-MCL) — exact port of the official code. Backbone of the main result (65.37).

beats branch of the original main.py set_model/train/validate:
  features = BEATsTransferLearningModel(...)(images)          # extract_features → (B, T', 768)   (models/BEATs/beats.py)
  features = features.mean(dim=1)                             # token mean (main.py:526)  NB: PAFA applies a per-token Linear, then averages — different
  output   = classifier(features)                             # nn.Linear(768, n_cls=3) (main.py:345)
  ssl_out  = ssl_classifier(features[ssl_label >= 0])          # nn.Linear(768, 2) (:368; placed after a projector with --ssl_separate_training)
  loss     = (1 - ssl_weight) * BCEWithLogits(output, labels) + ssl_weight * CE(ssl_out, ssl_label)   (:563-586)
SpecAugment: unless --nospec, SpecAugment(args) is applied to the BEATs internal fbank (:319-322). No script is released → default off (assumption).
No AMP (no autocast in main.py) → trainer precision 32.
Evaluation (validate): test cycles only → sigmoid > threshold → 4-class via the conversion.py rules → Sp/Se/Score.

Toolkit wiring: training batch = (waveform (B,L), 3-label float (B,3), ssl_label long (B)) — PLAdapter passes batch[2] as meta_labels.
Validation batch = (waveform, 4-class long) — BCE after the 4→3 conversion is val_loss; the returned logits are a 3-sigmoid→4-class one-hot (B,4)
(to match the argmax in ICBHIScore; the metric registry is unchanged).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from rsc_toolkit.methods.pcmcl import PMHead, pcmcl_training_loss
from rsc_toolkit.methods.pcmcl_labelspace import get_label_space

from .base import BaseRSCModel
from .beats_pafa import BEATsTransferLearningModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)


class _BEATsFbankSpecAugment(nn.Module):
    """Applies the original SpecAugment(args) to the BEATs fbank [B,1,T,F]: the original masks [B,F,T] via squeeze→transpose."""

    def __init__(self, policy: str = "icbhi_ast_sup", mask: str = "mean"):
        super().__init__()
        from rsc_toolkit.augmentations.specaugment import SpecAugment

        self.aug = SpecAugment(policy=policy, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, T, F)
        y = self.aug(x.squeeze(1).transpose(1, 2))       # (B, F, T)
        return y.transpose(1, 2).unsqueeze(1)


@ModelFactory.register_model_class("beats_pcmcl")
class BEATsPCMCL(BaseRSCModel, nn.Module):
    accepts_meta = ("pcmcl_ssl_label",)  # training batch[2] = ssl_label long -1/0/1 (PCMCLDataModule)

    def __init__(
        self,
        n_cls: int = 3,                        # 3-label (default/noconcat_multi) | 4 (ablation concat_single, CE)
        label_mode: str = "multilabel",        # multilabel | single
        label_space: str = "icbhi4",           # icbhi4 (3 attributes) | sprsound7 (6 attributes) — n_cls must equal the attribute count
        pretrained_path: Optional[str] = None,
        enable_ssl: bool = True,               # PM head (false for the Table 3 'Concat+Multi' ablation)
        ssl_weight: float = 0.1,               # paper α=0.1 ↔ code --ssl_weight (argparse default 0.5) — assumption
        ssl_separate_training: bool = False,
        ssl_projection_dim: int = 256,
        loss_combine: str = "code",            # code: (1-w)·main + w·ssl | paper: main + w·ssl
        threshold: float = 0.5,
        specaug: bool = False,                 # original default is on (unless --nospec); no script released, so nospec is assumed as in PAFA
        specaug_policy: str = "icbhi_ast_sup",
        specaug_mask: str = "mean",
    ):
        super().__init__()
        if label_mode not in {"multilabel", "single"}:
            raise ValueError(f"label_mode must be multilabel|single, got {label_mode!r}")
        self.ls = get_label_space(label_space)
        if label_mode == "multilabel" and n_cls != self.ls.n_attrs:
            raise ValueError(
                f"for multilabel, n_cls must equal the number of attributes in the label space — "
                f"{self.ls.name} has n_cls={self.ls.n_attrs} ({', '.join(self.ls.attr_names)}), got {n_cls}"
            )
        spec_transform = _BEATsFbankSpecAugment(specaug_policy, specaug_mask) if specaug else None
        self.backbone = BEATsTransferLearningModel(num_target_classes=n_cls, model_path=pretrained_path, spec_transform=spec_transform)
        self.final_feat_dim = self.backbone.final_feat_dim
        self.classifier = nn.Linear(self.final_feat_dim, n_cls)
        self.label_mode = label_mode
        self.enable_ssl = bool(enable_ssl) and label_mode == "multilabel"
        self.pm_head = PMHead(self.final_feat_dim, separate=ssl_separate_training, projection_dim=ssl_projection_dim) if self.enable_ssl else None
        self.ssl_weight, self.loss_combine, self.threshold = float(ssl_weight), loss_combine, float(threshold)

    def forward_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        feats = self.backbone(waveform)                     # (B, T', 768)
        if feats.dim() == 3:
            feats = feats.mean(dim=1)                      # (B, 768)
        return feats

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        features = self.forward_features(audio_batch)
        logits = self.classifier(features)
        if self.label_mode == "single":
            return logits, loss_func(logits, labels)
        if labels.dtype.is_floating_point:                 # training batch: 3-label (+ PM label)
            loss, _, _ = pcmcl_training_loss(logits, features, labels, loss_func, self.pm_head,
                                             meta_labels, self.ssl_weight, self.loss_combine)
            return logits, loss
        if self.training:  # contract violation (datasets/contract.py pcmcl_ssl_label) — do not silently train a mismatched combination
            raise TypeError(
                f"PC-MCL label_mode=multilabel expects float (B, 3) multi-label training batches, got {labels.dtype} — "
                "if the pcmcl datamodule mode is concat_single, set model.params.label_mode=single."
            )
        # validation batch: 4-class long → 3-label BCE (criterion(output, labels) in the original validate) + 4-class one-hot prediction
        loss = loss_func(logits, self.ls.encode(labels))
        return self.ls.logits_to_onehot(logits, self.threshold), loss
