# Portions of this file are adapted from patch-mix_contrastive_learning (https://github.com/raymin0223/patch-mix_contrastive_learning); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Loss functions of the Patch-Mix paper (Interspeech 2023), a faithful port of the official code.

Sources:
- Paper: Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on
  Respiratory Sound Classification (https://arxiv.org/abs/2305.14032)
- Official code: https://github.com/raymin0223/patch-mix_contrastive_learning
  method/patchmix.py, method/patchmix_cl.py (MIT, Copyright (c) Sangmin Bae)

Non-structural differences from the original code (the math is identical):
- The original code hardcodes `.cuda()` on the mask tensors. Here the device of the input tensor
  is used, so that the golden comparison can run in CPU tests.
- The original PatchMixConLoss.forward reads negative_pair from the `args` object.
  Here it is the constructor argument `negative_pair` (default 'all' = the paper's main experiment).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMixLoss(nn.Module):
    """lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b). Patch mixing is done in the model."""

    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion

    def forward(self, pred, y_a, y_b, lam):
        loss = lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
        return loss


class PatchMixConLoss(nn.Module):
    def __init__(self, temperature: float = 0.06, negative_pair: str = "all"):
        super().__init__()
        if negative_pair not in {"all", "diff_label"}:
            raise ValueError(f"negative_pair must be 'all' or 'diff_label', got {negative_pair!r}")
        self.temperature = temperature
        self.negative_pair = negative_pair

    def forward(self, projection1, projection2, labels_a, labels_b, lam, index):
        device = projection1.device
        batch_size = projection1.shape[0]
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        anchor_dot_contrast = torch.div(torch.matmul(projection2, projection1.T), self.temperature)

        mask_a = torch.eye(batch_size, device=device)
        mask_b = torch.zeros(batch_size, batch_size, device=device)
        mask_b[torch.arange(batch_size).unsqueeze(1), index.view(-1, 1)] = 1

        mask = lam * mask_a + (1 - lam) * mask_b

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # for numerical stability

        exp_logits = torch.exp(logits)
        if self.negative_pair == "diff_label":
            labels_a = labels_a.contiguous().view(-1, 1)
            logits_mask = torch.ne(labels_a, labels_a.T).to(device) + (mask_a.bool() + mask_b.bool())
            # The original code (method/patchmix_cl.py) does this in place (`exp_logits *= ...`), but exp's
            # backward saves its own output, so backward on this path always fails with
            # RuntimeError there (the only intentional difference from the original code; values are identical).
            exp_logits = exp_logits * logits_mask.float()

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size)

        loss = loss.mean()
        return loss
