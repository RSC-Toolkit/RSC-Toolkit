"""
M2D ICBHI/SPRSound 미세조정 모델 — nttcslab/m2d app/icbhi_sprs (scl_icbhi2017 패치본) 충실 재현.

참조: patch_scl_icbhi2017.diff — RT_LMS_M2D(encoder_only, freeze_embed): encode_lms → mean(dim=1) → Linear(3840, n_cls);
app_main.py: torchaudio MelSpectrogram(400/160, 80 mel, 50–8000) → AmplitudeToDB → 배치 min-max → [SpecAug(scl)] →
Standardize(ICBHI 0.3671/0.2391, SPRS 0.2000/0.2094). ce.py: split_iter K — 배치 전체를 transform 한 뒤 D=B//K 개씩
K 번 forward/backward(손실 합 누적), optimizer.step 1회; 나머지 B-K·D 개는 버림(val 도 동일, val 은 부분배치별 transform).
split_iter 실행은 trainer/adapters/split_iter_adapter.py 가 담당하고, 이 모델은 transform·forward 만 제공한다.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .m2d import M2DEncoder, load_m2d_checkpoint
from rsc_toolkit.augmentations.specaugment_torchaudio import SpecAugmentTA
from rsc_toolkit.processors.sclmeta_mel import batch_minmax, standardize

logger = logging.getLogger(__name__)


@ModelFactory.register_model_class("m2d")
class M2DFinetune(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        pretrained_path: Optional[str] = None,   # checkpoint-*.pth (None 이면 랜덤 초기화)
        input_size=(80, 200),
        patch_size=(16, 4),
        freeze_embed: bool = True,               # --freeze_embed (patch_embed 동결)
        std_mean: float = 0.3671,
        std_std: float = 0.2391,
        freq_mask: int = 20, time_mask: int = 40, freq_stripes: int = 2, time_stripes: int = 2, specaug_p: float = 1.0,
    ):
        super().__init__()
        self.encoder = M2DEncoder(img_size=tuple(input_size), patch_size=tuple(patch_size))
        if pretrained_path is not None:
            from rsc_toolkit.paths import resolve_pretrained

            load_m2d_checkpoint(self.encoder, resolve_pretrained(explicit=pretrained_path))  # RSCTK_PRETRAINED_ROOT 인정
        if freeze_embed:
            for p in self.encoder.patch_embed.parameters():
                p.requires_grad = False
        self.feature_d = self.encoder.embed_dim * self.encoder.grid_size()[0]  # 768 * 5
        self.linear = nn.Linear(self.feature_d, n_cls)  # 원본 RT_LMS_M2D.linear
        self.std_mean, self.std_std = std_mean, std_std
        self.specaug = SpecAugmentTA(freq_mask, time_mask, freq_stripes, time_stripes, specaug_p)

    @staticmethod
    def _as_4d(x):
        return x.unsqueeze(1) if x.dim() == 3 else x

    def train_transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return standardize(self.specaug(batch_minmax(self._as_4d(x))), self.std_mean, self.std_std)

    def val_transform(self, x: torch.Tensor) -> torch.Tensor:
        return standardize(batch_minmax(self._as_4d(x)), self.std_mean, self.std_std)

    def forward(self, x_transformed: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.encode_lms(x_transformed)  # (B, T', 3840)
        return self.linear(feats.mean(dim=1))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """split_iter 없이(K=1) 쓰는 경로. K>1 은 SplitIterAdapter 가 train/val_transform + forward 를 직접 호출한다."""
        x = self.train_transform(audio_batch) if self.training else self.val_transform(audio_batch)
        logits = self.forward(x)
        return logits, loss_func(logits, labels)
