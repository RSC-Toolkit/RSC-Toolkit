# Portions of this file are adapted from patch-mix_contrastive_learning (https://github.com/raymin0223/patch-mix_contrastive_learning); that code is under the MIT license.
# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class SpecAugmentConfig:
    enabled: bool = False
    policy: str = "icbhi_ast_sup"
    mask: str = "mean"  # "mean" | "zero"


class SpecAugment(nn.Module):
    """
    SpecAugment implementation adapted from the legacy training script.
    Applies time/frequency masking to mel/fbank features.
    """

    def __init__(self, policy: str, mask: str = "mean"):
        super().__init__()
        self.policy = policy
        self.mask = mask

        if policy == "LB":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif policy == "LD":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif policy == "SM":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif policy == "SS":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        elif policy == "icbhi_sup":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 0, 20, 2, 50, 1.0, 2
        elif policy == "icbhi_ast_sup":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 0, 48, 2, 160, 1.0, 2
        elif policy == "none":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 0, 0, 0, 0, -1.0, 0
        else:
            raise ValueError(f"Unknown SpecAugment policy: {policy}")

    def _to_bft(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[int]]:
        if x.dim() == 2:
            return x.unsqueeze(0), 2
        if x.dim() == 3:
            return x, 3
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected channel=1 for 4D input, got shape {tuple(x.shape)}")
            return x.squeeze(1), 4
        raise ValueError(f"Expected 2D/3D/4D tensor, got {tuple(x.shape)}")

    def _from_bft(self, x: torch.Tensor, original_dim: Optional[int]) -> torch.Tensor:
        if original_dim == 2:
            return x.squeeze(0)
        if original_dim == 4:
            return x.unsqueeze(1)
        return x

    def _mask_value(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask == "mean":
            return x.mean()
        if self.mask == "zero":
            return x.new_tensor(0.0)
        raise ValueError(f"Unknown mask type: {self.mask}")

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.F <= 0 or self.m_F <= 0:
            return x
        mask_value = self._mask_value(x)
        v = x.shape[1]
        for _ in range(self.m_F):
            # Draw randint even when f==0 to match the reference implementation's RNG consumption
            # (the reference assigns to an empty slice: same values, RNG stream stays in lockstep)
            f = int(np.random.uniform(0, self.F))
            f0 = random.randint(0, max(0, v - f))
            x[:, f0 : f0 + f, :] = mask_value
        return x

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.T <= 0 or self.m_T <= 0:
            return x
        mask_value = self._mask_value(x)
        tau = x.shape[2]
        for _ in range(self.m_T):
            t = int(np.random.uniform(0, self.T))
            t0 = random.randint(0, max(0, tau - t))
            x[:, :, t0 : t0 + t] = mask_value
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bft, original_dim = self._to_bft(x)
        if self.p < torch.randn(1).item():
            return self._from_bft(x_bft, original_dim)

        out = x_bft.clone()
        if self.W > 0:
            # Time warp omitted; legacy policies used W=0 for ICBHI AST.
            pass
        out = self._freq_mask(out)
        out = self._time_mask(out)
        return self._from_bft(out, original_dim)
