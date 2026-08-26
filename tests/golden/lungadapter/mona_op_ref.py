"""
Mona(arXiv 2311.15010, [22] in LungAdapter) 의 MonaOp 참조 구현 — LungAdapter Attention-Adapter 식(6)-(7) conv 부분의 골든.

출처: https://github.com/Leiyi-Hu/mona  Swin-Transformer-Classification/mmcls/models/backbones/swin_transformer_mona.py (class MonaOp)
      라이선스: MIT (원 파일 헤더 "Copyright (c) 2021 Microsoft, Licensed under The MIT License"; Mona 저장소 LICENSE 도 MIT)
      사본: outputs/reference/papers/lungadapter/mona_ref.py:27-48 (2026-08-26 취득) 에서 MonaOp 클래스만 그대로 수록(mmcv 의존 제거).
"""
import torch.nn as nn


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x
