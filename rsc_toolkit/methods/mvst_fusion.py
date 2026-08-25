"""
MVST(ICASSP 2024)의 게이티드 융합 — 공식 gated_fusion.py / fusion.py 충실 이식.

FuseMLP: 뷰별 768x768 게이트 행렬 W_i(kaiming_normal_), G_i = sigmoid(x_i @ W_i), F = Σ G_i * x_i, fc Linear(768, n_cls)
(weight N(0, 0.01), bias 0). 논문 식 (5)는 시그모이드 안에 전 뷰 합·토큰 단위지만 코드는 뷰별 독립 게이트·풀링 특징 — P0 는 코드 기준.

fusion.py 프로토콜: seed 1 · 특징은 save_features.py 로 덤프한 (train: 셔플·drop_last·SpecAugment 켠 1 pass / test: 순서대로)
768-d 뷰 특징 · CE 가중치 = 클래스 빈도 **비율**(counts/sum; train·test 각각) · AdamW(lr 1e-3, wd 1e-4) ·
CosineAnnealingLR(T_max=5, eta_min=1e-15) epoch step · 100 ep · bs 8 shuffle · autocast(GradScaler 없음) ·
best = test Score 최고 && Se > 5. 원본은 best 모델을 저장하지 않는다(출력만) — 여기서는 저장한다.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FuseMLP(nn.Module):
    def __init__(self, num_views: int = 5, num_classes: int = 4, representation_size: int = 768):
        super().__init__()
        self.num_classes = num_classes
        ws = []
        for _ in range(num_views):
            W = torch.empty(representation_size, representation_size, requires_grad=True)
            nn.init.kaiming_normal_(W)
            ws.append(nn.Parameter(W))
        # 원본 속성명 W1..W5 유지
        for i, W in enumerate(ws, start=1):
            setattr(self, f"W{i}", W)
        self.num_views = num_views
        self.fc = nn.Linear(representation_size, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, *xs):
        F = 0
        for i, x in enumerate(xs, start=1):
            F = F + torch.sigmoid(x @ getattr(self, f"W{i}")) * x
        return self.fc(F)


def get_score(hits, counts):
    sp = hits[0] / counts[0] * 100
    se = sum(hits[1:]) / sum(counts[1:]) * 100
    return sp, se, (sp + se) / 2


def class_ratio_weights(labels: torch.Tensor, n_cls: int = 4) -> torch.Tensor:
    """fusion.py:249-251 — counts / counts.sum() (다수 클래스가 큰 가중; 원본 quirk)."""
    counts = torch.bincount(labels, minlength=n_cls).float()
    return counts / counts.sum()


def train_fusion(train_feats: Sequence[torch.Tensor], train_labels: torch.Tensor,
                 test_feats: Sequence[torch.Tensor], test_labels: torch.Tensor,
                 epochs: int = 100, batch_size: int = 8, lr: float = 1e-3, weight_decay: float = 1e-4,
                 t_max: int = 5, eta_min: float = 1e-15, seed: int = 1, min_se: float = 5.0,
                 device: str = "cuda", amp: bool = True) -> Dict:
    """fusion.py main() 의 학습·평가 루프. 반환: best {S_p, S_e, icbhi_score, epoch, state_dict}."""
    torch.manual_seed(seed); np.random.seed(seed)
    import random
    random.seed(seed)
    n_cls = int(train_labels.max().item()) + 1
    model = FuseMLP(num_views=len(train_feats), num_classes=n_cls).to(device)
    loss_train = nn.CrossEntropyLoss(weight=class_ratio_weights(train_labels, n_cls).to(device))
    loss_test = nn.CrossEntropyLoss(weight=class_ratio_weights(test_labels, n_cls).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    train_ds = torch.utils.data.TensorDataset(*train_feats, train_labels)
    test_ds = torch.utils.data.TensorDataset(*test_feats, test_labels)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    best = {"S_p": 0.0, "S_e": 0.0, "icbhi_score": 0.0, "epoch": None, "state_dict": None}
    use_amp = amp and str(device).startswith("cuda")
    for epoch in range(1, epochs + 1):
        model.train()
        for *feats, labels in train_loader:
            feats = [f.to(device) for f in feats]; labels = labels.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_train(model(*feats), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        hits, counts = [0.0] * n_cls, [0.0] * n_cls
        with torch.no_grad():
            for *feats, labels in test_loader:
                feats = [f.to(device) for f in feats]; labels = labels.to(device)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    output = model(*feats)
                preds = output.argmax(1)
                for p, t in zip(preds.tolist(), labels.tolist()):
                    counts[t] += 1
                    if p == t:
                        hits[t] += 1
        sp, se, sc = get_score(hits, counts)
        if sc > best["icbhi_score"] and se > min_se:
            best = {"S_p": sp, "S_e": se, "icbhi_score": sc, "epoch": epoch,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        scheduler.step()
        logger.info("fusion epoch %d: Sp %.2f Se %.2f Score %.2f (best %.2f @%s)", epoch, sp, se, sc, best["icbhi_score"], best["epoch"])
    return best
