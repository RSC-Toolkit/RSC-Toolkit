"""
SG-SCL 논문(ICASSP 2024)의 학습 방법 — 공식 코드 충실 이식 (P0).

출처:
- 논문: Stethoscope-Guided Supervised Contrastive Learning for Cross-Domain
  Adaptation on Respiratory Sound Classification (https://arxiv.org/abs/2312.09603)
- 공식 코드: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
  method/mcl.py (MetaCL), main.py train() 의 domain_adaptation/domain_adaptation2 분기.

원본과의 비구조적 차이 (수치 영향 없음):
- .cuda() 대신 입력 텐서의 device 를 따른다.
- 원본 SpecAugment(util/augmentation.py)는 in-place 로 배치를 덮어쓰기 때문에
  train() 의 두 번째 `args.transforms(images)` 는 첫 뷰의 마스크 **위에** 추가
  마스킹한다(뷰2 = 뷰1 + 추가 마스크; 마스크값 mean 도 뷰1 마스킹 후 평균).
  AMP autocast 가 conv 입력을 half 로 복사해 두므로 backward 는 뷰1/뷰2 값
  그대로 진행된다. 여기서는 같은 semantics 를 함수형으로 재현한다
  (sgscl_training_losses 의 view2 = specaug(view1)).
- 원본의 미사용 projector(도메인 분기에서 생성만 되고 손실에 안 쓰이며 MA 로
  classifier[1] 스냅샷과 섞이는 것)는 학습에 영향이 없어 재현하지 않는다
  (HANDOFF 문서 판정 참고).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaCL(nn.Module):
    """원본 method/mcl.py — SupCon 형식이되 mask 를 meta 라벨(청진기) 동일 여부로 구성."""

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
    """원본 main.py:382-415 의 target_type 분기 그대로 (deepcopy(t.detach()) ≡ detach().clone())."""
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
    if target_type == "project1_project2block":  # 논문/스크립트 기본값
        return projector(feat1), projector(feat2).detach().clone()
    if target_type == "project_block_all":
        return projector(feat1).detach().clone(), projector(feat2).detach().clone()
    raise ValueError(f"Unknown target_type: {target_type!r}")


def apply_specaug_paper(specaug: nn.Module, x_paper: torch.Tensor) -> torch.Tensor:
    """
    페이퍼 레이아웃 (B, time, freq) 배치에 SpecAugment 1회 (배치 전체에 마스크 1벌).

    원본 SpecAugment.forward 는 [B,1,T,F]→[B,F,T] 로 뒤집어 마스킹 후 되돌린다.
    우리 포팅본(augmentations/specaugment.py)은 (B,F,T) 를 받으므로 같은 래핑을 한다.
    """
    return specaug(x_paper.transpose(-2, -1)).transpose(-2, -1)


def sgscl_training_losses(
    *,
    method: str,                 # "ce" | "dat" | "sgscl"
    paper_features,              # (B, T, F) 페이퍼 레이아웃 → (B, feat_dim) 특징
    x_paper: torch.Tensor,
    specaug,                     # None 이면 증강 없이 (원본 validate 경로에는 없음)
    classifier: nn.Module,
    domain_classifier,           # dat: 도메인 분류 헤드 / sgscl: Projector
    metacl,                      # sgscl 일 때 MetaCL
    loss_func,
    labels: torch.Tensor,
    meta_labels,
    alpha: float,                # 손실 가중 (--alpha, 기본 1.0)
    grl_alpha: float,            # GRL 계수 — 공식 레시피에선 항상 1.0 (아래 참고)
    target_type: str,
):
    """
    원본 main.py train() 의 method=='ce' 아래 세 분기와 동일한 손실 구성.

    GRL 계수: 원본은 p = (idx + epoch*len(loader))/epochs + 1/len(loader),
    alpha = 2/(1+exp(-10p))-1 로 매 반복 계산하지만, 공식 DAT 레시피
    (bs8→len=517, epochs=50, epoch 는 1부터)에서는 최솟값 p≈10.3 이라
    exp(-103) 이 언더플로해 **모든 반복에서 정확히 1.0** 이다. 그래서 상수로 둔다.
    """
    view1 = apply_specaug_paper(specaug, x_paper) if specaug is not None else x_paper
    feats1 = paper_features(view1)
    logits = classifier(feats1)
    class_loss = loss_func(logits, labels)

    if method == "ce":
        return logits, class_loss

    if meta_labels is None:
        raise ValueError(
            f"method={method!r} 에는 meta_labels 가 필요하다 "
            "(datamodule 의 meta_label_mode 설정을 확인)"
        )

    if method == "dat":
        # 지연 임포트: models/__init__ 의 일괄 등록과 순환을 피한다
        # (models.ast_sgscl → methods.sgscl → models.grl → models/__init__)
        from rsc_toolkit.models.grl import ReverseLayerF

        domain_feats = ReverseLayerF.apply(feats1, grl_alpha)
        meta_loss = F.cross_entropy(domain_classifier(domain_feats), meta_labels)  # 원본 criterion2: 비가중 CE
        return logits, class_loss + alpha * meta_loss

    if method == "sgscl":
        # 원본은 in-place SpecAugment 라 두 번째 호출이 뷰1 위에 마스크를 쌓는다.
        view2 = apply_specaug_paper(specaug, view1) if specaug is not None else x_paper
        feats2 = paper_features(view2)
        proj1, proj2 = sgscl_projections(feats1, feats2, domain_classifier, target_type)
        meta_loss = metacl(proj1, proj2, meta_labels)
        return logits, class_loss + alpha * meta_loss

    raise ValueError(f"Unknown sgscl method: {method!r}")
