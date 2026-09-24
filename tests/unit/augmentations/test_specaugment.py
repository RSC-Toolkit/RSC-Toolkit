# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
import random

import numpy as np
import torch

from rsc_toolkit.augmentations.specaugment import SpecAugment


def test_specaugment_policy_none_no_change():
    aug = SpecAugment(policy="none")
    x = torch.randn(1, 1, 4, 4)
    out = aug(x)
    assert torch.allclose(out, x)


def test_specaugment_zero_mask_applies(monkeypatch):
    aug = SpecAugment(policy="icbhi_ast_sup", mask="zero")
    aug.p = 1e9

    monkeypatch.setattr(np.random, "uniform", lambda low, high: high - 1)
    monkeypatch.setattr(random, "randint", lambda low, high: low)

    x = torch.ones(2, 1, 8, 10)
    out = aug(x)
    assert out.shape == x.shape
    assert (out == 0).any()
