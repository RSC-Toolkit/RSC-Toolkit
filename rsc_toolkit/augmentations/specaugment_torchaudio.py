"""
SCL-metadata 저장소의 SpecAugment — augmentations.py 그대로 (torchaudio iid 마스킹, 값 0).

quirk 유지: `if self.p > torch.randn(1)` 게이트 — p=1.0 이면 실제 적용률 ≈ 84% (patch-mix 계열과 같은 버그).
입력 (B, 1, F, T); FrequencyMasking(20)×2 + TimeMasking(40)×2, iid_masks=True → 샘플별 독립 마스크.
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
