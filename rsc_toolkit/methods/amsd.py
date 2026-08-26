"""
AMS-D(ICASSP 2026, CcnNnn/AMS-D) 의 학습 손실 — 공식 custom_loss.py Custom_Loss 의 충실 이식 (P0).

    L = L_task + α·L_distill + β·L_mask + γ·L_recon                (논문 식 4; custom_loss.py:45)
      L_task    = (-y_onehot · log_softmax(s)).mean(0).sum()      = 비가중 CE (custom_loss.py:27-28)
      L_mask    = 같은 식을 TWM 보조 로짓에 (weighted_output None 이면 0)  (custom_loss.py:31-35)
      L_distill = KLDivLoss(batchmean)(log_softmax(s/τ), softmax(t/τ)) · τ²  (custom_loss.py:38-41, 'soft')
                  ※ t_output 은 main.py:322/348 이 저장한 **이미 softmax 된 확률(epoch 0 은 one-hot)** 이라 teacher 에 softmax 가
                    두 번 걸린다(이중 softmax; epoch 0 실효 teacher = [0.475, 0.175, 0.175, 0.175]). 공식 동작이므로 기본값으로 유지하고,
                    논문 식 1(KL(q_t ∥ q_{t-1}), q 는 softmax 출력) 의 문자 그대로 해석은 teacher_is_prob=True 플래그 뒤에 둔다
                    (가정 — 저자 확인 전, docs/AUTHOR_QUESTIONS #7 계열).
      L_recon   = 호출측이 계산한 MSE 스칼라 (epoch 0 은 0)
    'hard' 는 CE(s, argmax t) (custom_loss.py:42-43) — 공식 실행은 'soft'.
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
        # False(기본, 공식): softmax(t/τ) 를 한 번 더 적용. True(가정): t 를 확률로 보고 그대로 KL 타깃으로 쓴다.
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
