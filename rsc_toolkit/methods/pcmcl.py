"""
Shared pieces of PC-MCL (arXiv 2601.17080, wa976/PC-MCL): label conversion, PM (patient matching) head, loss combination.

Official code references:
- 4-class → 3-label: util/dataset_pcmcl.py _convert_to_multilabel (0→[1,0,0], 1→[0,1,0], 2→[0,0,1], 3→[0,1,1])
- 3-label prediction → 4-class: additive branch of method/conversion.py convert_original
  ([1,0,0] only → normal, c&w→both(3), c→1, w→2, all 0→normal). Predictions are sigmoid > threshold (main.py multilabel_accuracy).
- PM head: main.py set_model — ssl_classifier = Linear(768, 2); with --ssl_separate_training
  a projector Linear(768,256)→LayerNorm→ReLU→Linear(256,768) is placed in front.
- Loss combination: main.py train() — loss = (1 - ssl_weight) * main + ssl_weight * ssl  (code default, 'code')
  The paper's L_main + α·L_aux is available only via the 'paper' flag (different form).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 4-class(normal, crackle, wheeze, both) → 3-label(normal, crackle, wheeze)
LABEL4_TO_MULTILABEL = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
)


def label4_to_multilabel(y: torch.Tensor) -> torch.Tensor:
    """(B,) long 4-class → (B, 3) float. Same as the original _convert_to_multilabel."""
    return LABEL4_TO_MULTILABEL.to(y.device)[y.long()]


def multilabel_binary_to_label4(pred: torch.Tensor) -> torch.Tensor:
    """
    (B, 3) {0,1} → (B,) long 4-class. Same as the original conversion.py convert_original (additive):
      normal>0 & crackle==0 & wheeze==0 → 0 ; crackle>0 & wheeze>0 → 3 ; crackle>0 → 1 ; wheeze>0 → 2 ; otherwise (all 0) → 0
    """
    normal, crackle, wheeze = pred[:, 0] > 0, pred[:, 1] > 0, pred[:, 2] > 0
    out = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
    out = torch.where(crackle & wheeze, torch.full_like(out, 3), out)
    out = torch.where(crackle & ~wheeze, torch.full_like(out, 1), out)
    out = torch.where(wheeze & ~crackle, torch.full_like(out, 2), out)
    # normal-only and all-zero map to 0 (already 0). If normal and an adventitious sound are both on, the adventitious sound wins, as the rules above express.
    return out


def multilabel_logits_to_label4(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """3-logit → sigmoid > threshold → 4-class (B,) long."""
    return multilabel_binary_to_label4((torch.sigmoid(logits) > threshold).to(torch.int64))


def multilabel_logits_to_onehot4(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    (B, 4) one-hot float to feed ICBHIScore (which assumes argmax over 4 classes) directly.
    Built so that argmax equals the converted 4-class prediction (without changing the metric registry).
    """
    y4 = multilabel_logits_to_label4(logits, threshold)
    return F.one_hot(y4, num_classes=4).to(logits.dtype)


class PMHead(nn.Module):
    """PM (same patient/recording or not) 2-class head. With separate=True, includes the projector of the original --ssl_separate_training."""

    def __init__(self, feat_dim: int = 768, separate: bool = False, projection_dim: int = 256):
        super().__init__()
        self.separate = bool(separate)
        if self.separate:
            self.ssl_projector = nn.Sequential(
                nn.Linear(feat_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, feat_dim),
            )
        else:
            self.ssl_projector = None
        self.ssl_classifier = nn.Linear(feat_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.ssl_projector is not None:
            features = self.ssl_projector(features)
        return self.ssl_classifier(features)


def combine_losses(main_loss: torch.Tensor, ssl_loss: Optional[torch.Tensor], weight: float, combine: str = "code") -> torch.Tensor:
    """
    combine='code'  : (1 - w) * main + w * ssl   (main.py:586, default)
    combine='paper' : main + w * ssl             (paper's L_main + α·L_aux; flag)
    If ssl_loss is None (batch without PM samples), main_loss is returned as-is (the original code also skips combining when ssl_mask.sum()==0).
    """
    if ssl_loss is None:
        return main_loss
    if combine == "code":
        return (1.0 - weight) * main_loss + weight * ssl_loss
    if combine == "paper":
        return main_loss + weight * ssl_loss
    raise ValueError(f"combine must be code|paper, got {combine!r}")


def pcmcl_training_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    loss_func: nn.Module,
    pm_head: Optional[PMHead],
    meta_labels: Optional[torch.Tensor],
    ssl_weight: float,
    combine: str,
):
    """
    Loss of the original train(): main = criterion(output, labels) (whole batch, PM samples included, no masking);
    ssl = CE(ssl_classifier(features[ssl_label >= 0]), ssl_label[mask]) (no combination if mask is empty).
    """
    main_loss = loss_func(logits, labels)
    ssl_loss = None
    if pm_head is not None and meta_labels is not None:
        mask = meta_labels >= 0
        if int(mask.sum()) > 0:
            ssl_out = pm_head(features[mask])
            ssl_loss = F.cross_entropy(ssl_out, meta_labels[mask].long())
    return combine_losses(main_loss, ssl_loss, ssl_weight, combine), main_loss, ssl_loss
