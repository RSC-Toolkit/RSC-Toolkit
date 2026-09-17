"""
SpecAugment from the SCL-metadata repository, ported as-is from augmentations.py (torchaudio iid masking, fill value 0).

Quirk kept: the `if self.p > torch.randn(1)` gate means p=1.0 gives an actual application rate of about 84% (same bug as the patch-mix family).
Input (B, 1, F, T); FrequencyMasking(20)×2 + TimeMasking(40)×2, iid_masks=True → independent masks per sample.
"""
import torch
import torch.nn as nn
from torchaudio import transforms as T


class SpecAugmentTA(nn.Module):
    def __init__(self, freq_mask: int = 20, time_mask: int = 40, freq_stripes: int = 2, time_stripes: int = 2, p: float = 1.0):
        super().__init__()
        self.p = p
        self.specaugment = nn.Sequential(
            *[T.FrequencyMasking(freq_mask_param=freq_mask, iid_masks=True) for _ in range(freq_stripes)],
            *[T.TimeMasking(time_mask_param=time_mask, iid_masks=True) for _ in range(time_stripes)],
        )

    def forward(self, audio):
        if self.p > torch.randn(1):
            return self.specaugment(audio)
        return audio
