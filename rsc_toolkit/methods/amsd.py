# Portions of this file are adapted from AMS-D (https://github.com/CcnNnn/AMS-D); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Training loss of AMS-D (ICASSP 2026, CcnNnn/AMS-D): a faithful port of Custom_Loss in the official custom_loss.py (official-code parity).

    L = L_task + α·L_distill + β·L_mask + γ·L_recon                (paper Eq. 4; custom_loss.py:45)
      L_task    = (-y_onehot · log_softmax(s)).mean(0).sum()      = unweighted CE (custom_loss.py:27-28)
      L_mask    = same formula on the TWM auxiliary logits (0 if weighted_output is None)  (custom_loss.py:31-35)
      L_distill = KLDivLoss(batchmean)(log_softmax(s/τ), softmax(t/τ)) · τ²  (custom_loss.py:38-41, 'soft')
                  Note: t_output is the **already-softmaxed probability (one-hot at epoch 0)** saved by main.py:322/348, so softmax is
                    applied to the teacher twice (double softmax; effective teacher at epoch 0 = [0.475, 0.175, 0.175, 0.175]). This is the official
                    behavior and is kept as the default; the literal reading of paper Eq. 1 (KL(q_t ∥ q_{t-1}), q = softmax output) is behind the
                    teacher_is_prob=True flag (Assumption: not confirmed by the authors).
      L_recon   = MSE scalar computed by the caller (0 at epoch 0)
    'hard' is CE(s, argmax t) (custom_loss.py:42-43); the official run uses 'soft'.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSDLoss(nn.Module):
    def __init__(
        self,
        distillation_type: str = "soft",
        tau: float = 1.0,
        alpha: float = 1.0,
        beta: float = 3e-2,
        gamma: float = 3e-1,
        teacher_is_prob: bool = False,
    ):
        super().__init__()
        if distillation_type not in ("soft", "hard"):
            raise ValueError(f"distillation_type must be soft|hard, got {distillation_type!r}")
        self.distillation_type = distillation_type
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        # False (default, official): apply softmax(t/τ) once more. True (assumption): treat t as probabilities and use it directly as the KL target.
        self.teacher_is_prob = bool(teacher_is_prob)
        self.kldivloss = nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def _soft_ce(logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        return (-targets_onehot * F.log_softmax(logits, dim=1)).mean(0).sum()

    def forward(
        self,
        s_output: torch.Tensor,
        t_output: torch.Tensor,
        weighted_output: Optional[torch.Tensor],
        targets_onehot: torch.Tensor,
        mse,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss = self._soft_ce(s_output, targets_onehot)
        if weighted_output is not None:
            weight_loss = self._soft_ce(weighted_output, targets_onehot)
        else:
            weight_loss = s_output.new_zeros(())
        if self.distillation_type == "soft":
            T = self.tau
            if self.teacher_is_prob:
                teacher = t_output if T == 1.0 else F.softmax(torch.log(t_output.clamp_min(1e-12)) / T, dim=1)
            else:
                teacher = F.softmax(t_output / T, dim=1)
            distillation_loss = self.kldivloss(F.log_softmax(s_output / T, dim=1), teacher) * (T * T)
        else:
            distillation_loss = F.cross_entropy(s_output, t_output.argmax(dim=1))
        total = loss + distillation_loss * self.alpha + weight_loss * self.beta + mse * self.gamma
        return total, loss, weight_loss, distillation_loss
