"""
Training method of the SG-SCL paper (ICASSP 2024): a faithful port of the official code (official-code parity).

Sources:
- Paper: Stethoscope-Guided Supervised Contrastive Learning for Cross-Domain
  Adaptation on Respiratory Sound Classification (https://arxiv.org/abs/2312.09603)
- Official code: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
  method/mcl.py (MetaCL), the domain_adaptation/domain_adaptation2 branches of main.py train().

Non-structural differences from the original code (no numerical effect):
- The device of the input tensor is used instead of .cuda().
- The original SpecAugment (util/augmentation.py) overwrites the batch in place, so
  the second `args.transforms(images)` in train() adds masking **on top of** the first view's
  mask (view2 = view1 + extra mask; the mask value mean is also taken after view1 masking).
  AMP autocast keeps a half-precision copy of the conv input, so backward proceeds with the view1/view2
  values as they were. The same semantics are reproduced functionally here
  (view2 = specaug(view1) in sgscl_training_losses).
- The original code's unused projector (created in the domain branch, never used in the loss, and mixed
  with the classifier[1] snapshot by MA) has no effect on training and is not reproduced.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaCL(nn.Module):
    """Original method/mcl.py: SupCon form, but the mask is built from whether meta labels (stethoscope) match."""

    def __init__(self, temperature: float = 0.06, weights=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights

    def forward(self, projection1, projection2, meta_labels=None):
        device = projection1.device
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if meta_labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        else:
            meta_labels = meta_labels.contiguous().view(-1, 1)
            mask = torch.eq(meta_labels, meta_labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * contrast_count, device=device).view(-1, 1), 0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss


def sgscl_projections(feat1, feat2, projector, target_type: str):
    """Same as the target_type branches of the original main.py:382-415 (deepcopy(t.detach()) ≡ detach().clone())."""
    if target_type == "project_flow_all":
        return projector(feat1), projector(feat2)
    if target_type == "representation_all":
        return feat1, feat2
    if target_type == "z1block_project":
        return feat1.detach().clone(), projector(feat2)
    if target_type == "z1_project2":
        return feat1, projector(feat2)
    if target_type == "project1block_project2":
        return projector(feat1).detach().clone(), projector(feat2)
    if target_type == "project1_r2block":
        return projector(feat1), feat2.detach().clone()
    if target_type == "project1_r2":
        return projector(feat1), feat2
    if target_type == "project1_project2block":  # paper/script default
        return projector(feat1), projector(feat2).detach().clone()
    if target_type == "project_block_all":
        return projector(feat1).detach().clone(), projector(feat2).detach().clone()
    raise ValueError(f"Unknown target_type: {target_type!r}")


def apply_specaug_paper(specaug: nn.Module, x_paper: torch.Tensor) -> torch.Tensor:
    """
    Applies SpecAugment once to a paper-layout (B, time, freq) batch (one mask set for the whole batch).

    The original SpecAugment.forward permutes [B,1,T,F]→[B,F,T], masks, and permutes back.
    The port (augmentations/specaugment.py) takes (B,F,T), so the same wrapping is done here.
    """
    return specaug(x_paper.transpose(-2, -1)).transpose(-2, -1)


def sgscl_training_losses(
    *,
    method: str,                 # "ce" | "dat" | "sgscl"
    paper_features,              # (B, T, F) paper layout → (B, feat_dim) features
    x_paper: torch.Tensor,
    specaug,                     # None means no augmentation (absent in the original validate path)
    classifier: nn.Module,
    domain_classifier,           # dat: domain classification head / sgscl: Projector
    metacl,                      # MetaCL for sgscl
    loss_func,
    labels: torch.Tensor,
    meta_labels,
    alpha: float,                # loss weight (--alpha, default 1.0)
    grl_alpha: float,            # GRL coefficient; always 1.0 in the official recipe (see below)
    target_type: str,
    cls_feat: str = "feat0",     # classifier input: "feat0"=x (official main.py:374) | "feat1"=GRL(x) (experimental)
    con_feat: str = "feat0",     # sgscl contrastive features: "feat0"=x (commit 6656460, default) | "feat1"=GRL(x) (GitHub HEAD 2ed1bac)
):
    """
    Same loss composition as the three branches under method=='ce' in the original main.py train().

    GRL coefficient: the original code computes p = (idx + epoch*len(loader))/epochs + 1/len(loader),
    alpha = 2/(1+exp(-10p))-1 every iteration, but in the official DAT recipe
    (bs8→len=517, epochs=50, epochs start at 1) the minimum is p≈10.3, so
    exp(-103) underflows and alpha is **exactly 1.0 in every iteration**. It is therefore kept constant.
    """
    if cls_feat not in {"feat0", "feat1"} or con_feat not in {"feat0", "feat1"}:
        raise ValueError(f"cls_feat/con_feat must be feat0|feat1, got {cls_feat!r}/{con_feat!r}")
    from rsc_toolkit.models.grl import ReverseLayerF  # deferred import (same reason as the dat branch below)

    view1 = apply_specaug_paper(specaug, x_paper) if specaug is not None else x_paper
    feats1 = paper_features(view1)
    # features of the original forward(training=True) = (x, GRL(x)); features[1] has the same values, only the gradient is scaled by −grl_alpha.
    cls_in = ReverseLayerF.apply(feats1, grl_alpha) if cls_feat == "feat1" else feats1
    logits = classifier(cls_in)
    class_loss = loss_func(logits, labels)

    if method == "ce":
        return logits, class_loss

    if meta_labels is None:
        raise ValueError(
            f"method={method!r} requires meta_labels "
            "(check the datamodule's meta_label_mode setting)"
        )

    if method == "dat":
        domain_feats = ReverseLayerF.apply(feats1, grl_alpha)
        meta_loss = F.cross_entropy(domain_classifier(domain_feats), meta_labels)  # original criterion2: unweighted CE
        return logits, class_loss + alpha * meta_loss

    if method == "sgscl":
        # The original SpecAugment is in place, so the second call stacks masks on top of view1.
        view2 = apply_specaug_paper(specaug, view1) if specaug is not None else x_paper
        feats2 = paper_features(view2)
        if con_feat == "feat1":
            # GitHub HEAD 2ed1bac (main.py:379-380): feat1/feat2 = features[1] = GRL(x); same values, only the
            # backbone gradient of the contrastive loss is reversed. Commit 6656460 uses features[0].
            f1, f2 = ReverseLayerF.apply(feats1, grl_alpha), ReverseLayerF.apply(feats2, grl_alpha)
        else:
            f1, f2 = feats1, feats2
        proj1, proj2 = sgscl_projections(f1, f2, domain_classifier, target_type)
        meta_loss = metacl(proj1, proj2, meta_labels)
        return logits, class_loss + alpha * meta_loss

    raise ValueError(f"Unknown sgscl method: {method!r}")
