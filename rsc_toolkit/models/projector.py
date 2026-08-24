"""
Patch-Mix / SG-SCL 계열 논문들이 공유하는 projection head — 공식 코드 충실 이식.

출처: https://github.com/raymin0223/patch-mix_contrastive_learning models/projector.py (MIT)
      (kaen2891/stethoscope-guided_supervised_contrastive_learning 도 동일 구조를 사용)
구조: Linear(in,in) - BatchNorm1d(in) - ReLU - Linear(in,out). apply_bn=False 면 BN 생략.
"""
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768, apply_bn: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x):
        return self.projector(x)
