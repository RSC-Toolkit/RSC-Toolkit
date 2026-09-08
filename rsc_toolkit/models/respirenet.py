"""
RespireNet(EMBC 2021)의 CNN 모델 — 공식 코드 충실 이식 (P0).

출처:
- 논문: RespireNet: A Deep Neural Network for Accurately Detecting Abnormal Lung Sounds in
  Limited Data Setting (https://arxiv.org/abs/2011.00196)
- 공식 코드: https://github.com/microsoft/RespireNet nets/network_cnn.py (MIT)

구조 (원본 그대로): torchvision resnet34(ImageNet) 의 fc 를
Dropout(0.5)-Linear(512,128)-ReLU-Dropout(0.5)-Linear(128,128)-ReLU 로 교체 + cls_fc Linear(128, n_cls).
입력은 3채널 magma 컬러맵 스펙트로그램 이미지 (B, 3, 192, 753) (processors/respirenet_image.py).
속성명 model_ft / cls_fc 를 유지해 저자 배포 models/ckpt_best.pkl 을 strict 로드한다.
fine_tune(block_layer): 원본은 eval.py 에서만 호출(train.py 에서는 주석) — 재현 시 미사용.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)


@ModelFactory.register_model_class("respirenet")
class RespireNetCNN(BaseRSCModel, nn.Module):
    accepts_meta = ("respirenet_chunk_index",)  # RespireNetDataModule 이 청크 인덱스를 붙인다 (모델은 무시)

    def __init__(self, n_cls: int = 4, drop_prob: float = 0.5, imagenet_pretrain: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if imagenet_pretrain else None
        self.model_ft = torchvision.models.resnet34(weights=weights)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Sequential(
            nn.Dropout(drop_prob), nn.Linear(num_ftrs, 128), nn.ReLU(True),
            nn.Dropout(drop_prob), nn.Linear(128, 128), nn.ReLU(True),
        )
        self.cls_fc = nn.Linear(128, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_fc(self.model_ft(x))

    def fine_tune(self, block_layer: int = 5) -> None:
        """원본 network_cnn.py:26-32 — children 0..block_layer 동결."""
        for idx, child in enumerate(self.model_ft.children()):
            if idx > block_layer:
                break
            for param in child.parameters():
                param.requires_grad = False

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)

    def load_reference_checkpoint(self, path) -> None:
        """저자 ckpt_best.pkl (model_ft.* / cls_fc.* 222 키) 를 strict 로드."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
        logger.info("RespireNet 저자 체크포인트 로드: %s", path)
