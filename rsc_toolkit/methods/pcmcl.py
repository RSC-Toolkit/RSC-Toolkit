"""
PC-MCL(arXiv 2601.17080, wa976/PC-MCL) 의 공통 조각 — 라벨 변환·PM(환자 매칭) 헤드·손실 결합.

공식 코드 근거:
- 4-class → 3-label: util/dataset_pcmcl.py _convert_to_multilabel (0→[1,0,0], 1→[0,1,0], 2→[0,0,1], 3→[0,1,1])
- 3-label 예측 → 4-class: method/conversion.py convert_original 의 additive 분기
  ([1,0,0] 만 normal, c&w→both(3), c→1, w→2, 전부 0→normal). 예측은 sigmoid > threshold (main.py multilabel_accuracy).
- PM 헤드: main.py set_model — ssl_classifier = Linear(768, 2); --ssl_separate_training 이면
  projector Linear(768,256)→LayerNorm→ReLU→Linear(256,768) 를 앞에 둔다.
- 손실 결합: main.py train() — loss = (1 - ssl_weight) * main + ssl_weight * ssl  (코드 기본, 'code')
  논문 식 L_main + α·L_aux 는 'paper' 플래그로만 둔다 (형태가 다름 — 정찰 verdict 참조).
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
    """(B,) long 4-class → (B, 3) float. 원본 _convert_to_multilabel 과 동일."""
    return LABEL4_TO_MULTILABEL.to(y.device)[y.long()]


def multilabel_binary_to_label4(pred: torch.Tensor) -> torch.Tensor:
    """
    (B, 3) {0,1} → (B,) long 4-class. 원본 conversion.py convert_original (additive) 그대로:
      normal>0 & crackle==0 & wheeze==0 → 0 ; crackle>0 & wheeze>0 → 3 ; crackle>0 → 1 ; wheeze>0 → 2 ; 그 외(전부 0) → 0
    """
    normal, crackle, wheeze = pred[:, 0] > 0, pred[:, 1] > 0, pred[:, 2] > 0
    out = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
    out = torch.where(crackle & wheeze, torch.full_like(out, 3), out)
    out = torch.where(crackle & ~wheeze, torch.full_like(out, 1), out)
    out = torch.where(wheeze & ~crackle, torch.full_like(out, 2), out)
    # normal-only 와 all-zero 는 0 (이미 0). normal 과 이상음이 같이 켜지면 이상음 우선 — 위 규칙이 그대로 표현한다.
    return out


def multilabel_logits_to_label4(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """3-logit → sigmoid > threshold → 4-class (B,) long."""
    return multilabel_binary_to_label4((torch.sigmoid(logits) > threshold).to(torch.int64))


def multilabel_logits_to_onehot4(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    ICBHIScore(argmax 4-class 가정) 에 그대로 넣기 위한 (B, 4) one-hot float.
    argmax 가 변환된 4-class 예측과 같아지도록 만든다 (메트릭 레지스트리 수정 없이).
    """
    y4 = multilabel_logits_to_label4(logits, threshold)
    return F.one_hot(y4, num_classes=4).to(logits.dtype)


class PMHead(nn.Module):
    """PM(같은 환자/녹음 여부) 2-class 헤드. separate=True 면 원본 --ssl_separate_training 의 projector 포함."""

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
    combine='code'  : (1 - w) * main + w * ssl   (main.py:586, 기본)
    combine='paper' : main + w * ssl             (논문 식 L_main + α·L_aux — 플래그)
    ssl_loss 가 None(PM 샘플 없는 배치) 이면 main_loss 그대로 (원본도 ssl_mask.sum()==0 이면 결합하지 않는다).
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
    원본 train() 의 손실: main = criterion(output, labels) (배치 전체 — PM 샘플 포함, 마스킹 없음);
    ssl = CE(ssl_classifier(features[ssl_label >= 0]), ssl_label[mask]) (mask 가 비면 결합 없음).
    """
    main_loss = loss_func(logits, labels)
    ssl_loss = None
    if pm_head is not None and meta_labels is not None:
        mask = meta_labels >= 0
        if int(mask.sum()) > 0:
            ssl_out = pm_head(features[mask])
            ssl_loss = F.cross_entropy(ssl_out, meta_labels[mask].long())
    return combine_losses(main_loss, ssl_loss, ssl_weight, combine), main_loss, ssl_loss
