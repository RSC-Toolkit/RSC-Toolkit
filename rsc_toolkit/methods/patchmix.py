"""
Patch-Mix 논문(Interspeech 2023)의 손실 함수 — 공식 코드 충실 이식.

출처:
- 논문: Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on
  Respiratory Sound Classification (https://arxiv.org/abs/2305.14032)
- 공식 코드: https://github.com/raymin0223/patch-mix_contrastive_learning
  method/patchmix.py, method/patchmix_cl.py (MIT, Copyright (c) Sangmin Bae)

원본과의 비구조적 차이 (수학은 동일):
- 원본은 마스크 텐서에 `.cuda()` 를 하드코딩했다. 여기서는 입력 텐서의 device 를
  따른다 — CPU 테스트가 가능해야 P0 골든 대조를 돌릴 수 있다.
- 원본 PatchMixConLoss.forward 는 `args` 객체에서 negative_pair 를 읽는다.
  여기서는 생성자 인자 `negative_pair` 로 받는다 (기본 'all' = 논문 본 실험).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMixLoss(nn.Module):
    """lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b). 패치 믹싱은 모델 쪽에 있다."""

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
            exp_logits *= logits_mask.float()

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size)

        loss = loss.mean()
        return loss
