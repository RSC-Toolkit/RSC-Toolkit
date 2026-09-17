# Portions of this file are ported from scl_icbhi2017 (https://github.com/ilyassmoummad/scl_icbhi2017), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
SCL-metadata (Moummad & Farrugia, ICASSP 2023; arXiv 2210.16192) — exact port of the official code.

Source: https://github.com/ilyassmoummad/scl_icbhi2017 (models.py, losses.py, ce.py, scl.py, mscl.py, hybrid.py, main.py)

Components (same as the original code):
- Encoder CNN6 (PANN ConvBlock5x5 x4, partial load of AudioSet Cnn6_mAP=0.343, no dropout) → 512-d.
  Blocks, pooling and partial loading are identical to the toolkit's sft_backbones.CNN6SFT (with input [B,1,F,T]: time mean → frequency max+mean).
- Batch transform (main.py:59-67, done inside the model): dB-mel (cached by the processor) → min-max over the whole batch → [SpecAugment (train)]
  → Standardize(0.3690, 0.0255).
- method
  ce     : Linear(512, n_cls), single view (SpecAug once), weighted CE (loss_func) — ce.py:22-28
  hybrid : two views → α·SupCon(proj1, proj2, y) + (1-α)·CE(cat[pred1,pred2], cat[y,y]) — losses.py:46-57, hybrid.py
  scl    : two views → SupCon(proj1, proj2, y), τ 0.06 — scl.py (pretraining, no evaluation, last-epoch state saved)
  mscl   : two views, 2 projectors → (1-λ)·SupCon(p1,p2, meta) + λ·SupCon(p1',p2', y), λ 0.75 — mscl.py:15-19
  linear : load encoder_ckpt and freeze it (requires_grad False + eval; BN statistics fixed); train only Linear(512, n_cls)
           on val_transform features (no SpecAug), weighted CE — scl.py:150-152, linear stage
- Projector: Linear(512,512)-ReLU-Linear(512,128), no BN (models.py:256-271, apply_bn=False).
- The SupCon loss has the same formula as MetaCL in methods/sgscl.py and reuses it (SimCLR when labels=None).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .projector import Projector
from .sft_backbones import CNN6SFT
from rsc_toolkit.augmentations.specaugment_torchaudio import SpecAugmentTA
from rsc_toolkit.methods.sgscl import MetaCL as SupConLoss
from rsc_toolkit.processors.sclmeta_mel import STANDARDIZE_MEAN, STANDARDIZE_STD, batch_minmax, standardize

logger = logging.getLogger(__name__)

METHODS = ("ce", "hybrid", "scl", "mscl", "linear")


@ModelFactory.register_model_class("cnn6_sclmeta")
class CNN6SCLMeta(BaseRSCModel, nn.Module):
    accepts_meta = ("sa", "sa_sprs")  # batch[2] = sex × age-group class (SCL-metadata; required for method=mscl)

    def __init__(
        self,
        n_cls: int = 4,
        method: str = "mscl",
        from_sl_official: bool = True,     # partial PANN AudioSet load (False for --scratch)
        out_dim: int = 128,
        temperature: float = 0.06,
        alpha: float = 0.5,                # hybrid: SupCon weight
        lam: float = 0.75,                 # mscl: class-SCL weight (meta gets 1-λ)
        freq_mask: int = 20, time_mask: int = 40, freq_stripes: int = 2, time_stripes: int = 2, specaug_p: float = 1.0,
        std_mean: float = STANDARDIZE_MEAN, std_std: float = STANDARDIZE_STD,
        encoder_ckpt: Optional[str] = None,  # linear: path to the final_state of the pretraining run ("model" key — final_state.py)
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.alpha, self.lam = alpha, lam
        self.std_mean, self.std_std = std_mean, std_std
        self.encoder = CNN6SFT()
        if from_sl_official and method != "linear":
            self.encoder.load_sl_official_weights()
        self.final_feat_dim = self.encoder.final_feat_dim
        self.specaug = SpecAugmentTA(freq_mask, time_mask, freq_stripes, time_stripes, specaug_p)
        self.supcon = SupConLoss(temperature=temperature)

        self.classifier = nn.Linear(self.final_feat_dim, n_cls) if method in ("ce", "hybrid", "linear") else None
        self.n_cls = int(n_cls)   # dummy logit width for scl/mscl pretraining (a fixed 4 would make train/acc@1 meaningless for 7-class)
        self.projector = Projector(self.final_feat_dim, out_dim, apply_bn=False) if method in ("hybrid", "scl", "mscl") else None
        self.projector2 = Projector(self.final_feat_dim, out_dim, apply_bn=False) if method == "mscl" else None

        if method == "linear":
            if encoder_ckpt is None:
                raise ValueError("method=linear requires encoder_ckpt (final_state.pth of the pretraining run)")
            state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            enc = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc, strict=True)  # strict, so any mismatch raises
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("Loaded and froze encoder for linear evaluation: %s", encoder_ckpt)

    # original linear stage: encoder.eval() fixed (BN running statistics not updated)
    def train(self, mode: bool = True):
        super().train(mode)
        if self.method == "linear":
            self.encoder.eval()
        return self

    # ----------------------------------------------------------- batch transform
    def _as_4d(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 3 else x  # (B, F, T) -> (B, 1, F, T)

    def _val_transform(self, x):
        return standardize(batch_minmax(self._as_4d(x)), self.std_mean, self.std_std)

    def _train_transform(self, x):
        with torch.no_grad():
            return standardize(self.specaug(batch_minmax(self._as_4d(x))), self.std_mean, self.std_std)

    def forward_features(self, x_transformed: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_transformed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError(f"method={self.method} has no classifier (pretraining only)")
        return self.classifier(self.forward_features(self._val_transform(x)))

    # ---------------------------------------------------------------- training
    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if not self.training or self.method == "linear":
            if self.classifier is None:
                # scl/mscl pretraining runs have no evaluation (limit_val_batches=0 in the config); defensive dummy
                z = torch.zeros(audio_batch.shape[0], self.n_cls, device=audio_batch.device)
                return z, torch.zeros((), device=audio_batch.device, requires_grad=self.training)
            logits = self.forward(audio_batch)
            return logits, loss_func(logits, labels)

        if self.method == "ce":
            logits = self.classifier(self.forward_features(self._train_transform(audio_batch)))
            return logits, loss_func(logits, labels)

        v1, v2 = self._train_transform(audio_batch), self._train_transform(audio_batch)
        f1, f2 = self.forward_features(v1), self.forward_features(v2)

        if self.method == "hybrid":
            p1, p2 = self.projector(f1), self.projector(f2)
            pred1, pred2 = self.classifier(f1), self.classifier(f2)
            loss = self.alpha * self.supcon(p1, p2, labels) + (1 - self.alpha) * loss_func(
                torch.cat([pred1, pred2], dim=0), torch.cat([labels, labels], dim=0))
            return pred1, loss
        if self.method == "scl":
            return torch.zeros(f1.shape[0], self.n_cls, device=f1.device), self.supcon(self.projector(f1), self.projector(f2), labels)
        # mscl
        if meta_labels is None:
            raise ValueError("method=mscl requires meta_labels (sa) — datamodule meta_label_mode=sa")
        p1, p2 = self.projector(f1), self.projector(f2)
        q1, q2 = self.projector2(f1), self.projector2(f2)
        loss = (1.0 - self.lam) * self.supcon(p1, p2, meta_labels) + self.lam * self.supcon(q1, q2, labels)
        return torch.zeros(f1.shape[0], self.n_cls, device=f1.device), loss
