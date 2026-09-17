# Portions of this file are adapted from patch-mix_contrastive_learning (https://github.com/raymin0223/patch-mix_contrastive_learning); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Projection head shared by the Patch-Mix / SG-SCL family of papers — exact port of the official code.

Source: https://github.com/raymin0223/patch-mix_contrastive_learning models/projector.py (MIT)
      (kaen2891/stethoscope-guided_supervised_contrastive_learning uses the same structure)
Structure: Linear(in,in) - BatchNorm1d(in) - ReLU - Linear(in,out). BN is omitted when apply_bn=False.
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
