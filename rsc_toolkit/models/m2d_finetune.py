"""
M2D ICBHI/SPRSound fine-tuning model — faithful reproduction of nttcslab/m2d app/icbhi_sprs (patched scl_icbhi2017).

Reference: patch_scl_icbhi2017.diff — RT_LMS_M2D(encoder_only, freeze_embed): encode_lms → mean(dim=1) → Linear(3840, n_cls);
app_main.py: torchaudio MelSpectrogram(400/160, 80 mel, 50–8000) → AmplitudeToDB → batch min-max → [SpecAug(scl)] →
Standardize(ICBHI 0.3671/0.2391, SPRS 0.2000/0.2094). ce.py: split_iter K — transform the whole batch, then run
K forward/backward passes of D=B//K samples each (losses accumulated), one optimizer.step; the remaining B-K·D samples are dropped. Validation transforms per sub-batch; the original also drops the remainder there,
while the toolkit evaluates it as an extra sub-batch (see the adapter).
split_iter execution is handled by trainer/adapters/split_iter_adapter.py; this model provides only transform and forward.
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
        pretrained_path: Optional[str] = None,   # checkpoint-*.pth (random init if None)
        input_size=(80, 200),
        patch_size=(16, 4),
        freeze_embed: bool = True,               # --freeze_embed (freeze patch_embed)
        std_mean: float = 0.3671,
        std_std: float = 0.2391,
        freq_mask: int = 20, time_mask: int = 40, freq_stripes: int = 2, time_stripes: int = 2, specaug_p: float = 1.0,
    ):
        super().__init__()
        self.encoder = M2DEncoder(img_size=tuple(input_size), patch_size=tuple(patch_size))
        if pretrained_path is not None:
            from rsc_toolkit.paths import resolve_pretrained

            load_m2d_checkpoint(self.encoder, resolve_pretrained(explicit=pretrained_path))  # honors RSCTK_PRETRAINED_ROOT
        if freeze_embed:
            for p in self.encoder.patch_embed.parameters():
                p.requires_grad = False
        self.feature_d = self.encoder.embed_dim * self.encoder.grid_size()[0]  # 768 * 5
        self.linear = nn.Linear(self.feature_d, n_cls)  # original RT_LMS_M2D.linear
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
        """Path used without split_iter (K=1). For K>1, SplitIterAdapter calls train/val_transform + forward directly."""
        x = self.train_transform(audio_batch) if self.training else self.val_transform(audio_batch)
        logits = self.forward(x)
        return logits, loss_func(logits, labels)
