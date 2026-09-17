"""
Lung-SRAD (Interspeech 2026, arXiv 2606.11922; RSC-Toolkit/Lung-SRAD d28d783) — DASS-medium(VMamba-S v2) + Spectral-Aware
Regularization (Gaussian blur hook) + Dual-Axis Patch-Mix CL, exact port of the official code.

Original main.py set_model/train (--model dass, model_size medium hard-coded):
  images (B,798,128) fbank(per-sample (x−mean)/(2·std)) → [SpecAugment icbhi_ast_sup] → Resize((1024,128)) → (B,1024,128)
  DASS.forward: unsqueeze(1) → transpose(2,3) → (B,1,128,1024) → VSSM(patch_embed → (B,96,32,256) → 4 stage → norm/avgpool 768) → head(768,4)
  method ce      : loss = CE(logits, y)                                   (= "DASS FT";  no --gaussian_blur → no blur blocks)
  method ce_blur : same loss, blur blocks {2,3}@stage2 enabled         (= "SAR"; original method ce + --gaussian_blur)
  method patchmix_cl (cl_mode dual, default): L = CE + α_t·L_PM^time + α_f·L_PM^freq  (main.py:310-351)
      proj1 = projector(features)  (target_type project_flow — gradient flows in the original path)
      independent forward for time and freq (independent index·λ) → mix features detached (=Eq.8 sg) → projector → PatchMixConLoss(τ 0.2, negative_pair all)
      cl_mode time|freq uses one axis only; cl_mode 2d is AST-style (the original has **no detach**, α=alpha) — kept as in the original code (detach_mix_2d=False).
  autocast (fp16) only, no GradScaler (main.py:305, 364-370; ast_models.py:361 @autocast applied twice) → autocast inside the model, trainer precision 32.

Pretraining: DASS_medium_v2.pth (Saurabhbhati/DASS v0.2, 'module.v.' prefix stripped, classifier.head excluded → only the 2 head keys are missing),
  patch_embed[0] is rebuilt as Conv2d(1,48,3,2,1) (imagenet_pretrain=False) and overwritten with the ckpt's single-channel weights (ast_models.py:328-336).

Quirks (code defaults kept; the paper definitions are available via flags):
  - mix_layout=as_code (default): patch-mix treats channel-first (B,96,32,256) as (B,H,W,C) → 'time' mixes the 32 frequency bins, 'freq' mixes the 96 channels,
    the λ denominators are 32/96, and the time axis is never mixed. as_paper: Eq.6-7 (time W=256 / frequency H=32). (see comments in backbone/vmamba/vssm.py)
  - AdamW (code misc.py) vs Adam (paper) → code.  drop_path 0.3 kept.  blur is a fixed operation that also applies at evaluation.
  - SpecAugment is applied by the datamodule from the **data config (specaugment.enabled, icbhi_ast_sup/mean, randn gate)** — not inside the model (to avoid applying it twice).
"""
from __future__ import annotations

import logging
import pathlib
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsc_toolkit.methods.patchmix import PatchMixConLoss

from .backbone.vmamba import MIX_LAYOUTS, build_dass_medium_vssm, cuda_kernel_available, load_dass_checkpoint
from .base import BaseRSCModel
from .factory import ModelFactory
from .projector import Projector

logger = logging.getLogger(__name__)

DASS_CKPT_CANDIDATES = (
    "pretrained_models/upstream/dass/DASS_medium_v2.pth",
    "outputs/reference/papers/lungsrad/DASS_medium_v2.pth",
)
METHODS = ("ce", "ce_blur", "patchmix_cl")
CL_MODES = ("dual", "time", "freq", "2d", "dual_seq")
TARGET_TYPES = ("grad_block", "grad_flow", "project_block", "project_flow")


@ModelFactory.register_model_class("dass_lungsrad")
class DASSLungSRAD(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        method: str = "ce",                       # ce (FT) | ce_blur (SAR) | patchmix_cl (Lung-SRAD and Table 2 variants)
        cl_mode: str = "dual",                    # patchmix_cl: dual | time | freq | 2d | dual_seq (variant from dead code)
        mix_layout: str = "as_code",              # as_code (original behavior) | as_paper (Eq.6-7)
        train_projector: bool = True,             # False = main.py behavior (projector not in the optimizer)
        gaussian_blur: Optional[bool] = None,     # None → method != 'ce' (original --gaussian_blur script default)
        blur_blocks: Sequence[int] = (2, 3),
        blur_stage: Sequence[int] = (2,),
        kernel_size: int = 5,
        sigma: float = 3.0,
        blur_strength: float = 1.0,
        mix_beta: float = 1.0,
        temperature: float = 0.2,
        negative_pair: str = "all",
        target_type: str = "project_flow",
        proj_dim: int = 768,
        alpha_time: float = 1.0,
        alpha_freq: float = 1.0,
        alpha: float = 1.0,                       # coefficient for cl_mode 2d (original --alpha)
        detach_mix_2d: bool = False,              # the original 2d branch has no detach (quirk); True applies sg as in Eq.8
        drop_path_rate: float = 0.3,
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,
        input_layout: str = "freq_time",          # toolkit fbank (B,128,798)
        resize_to: Tuple[int, int] = (128, 1024), # (F, T) — original Resize((1024,128)) on (T,F)
        amp: bool = True,                         # original autocast fp16, no GradScaler (trainer precision 32)
        ssm_backend: str = "auto",                # auto | torch | cuda_oflex
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if cl_mode not in CL_MODES:
            raise ValueError(f"cl_mode must be one of {CL_MODES}, got {cl_mode!r}")
        if mix_layout not in MIX_LAYOUTS:
            raise ValueError(f"mix_layout must be one of {MIX_LAYOUTS}, got {mix_layout!r}")
        if target_type not in TARGET_TYPES:
            raise ValueError(f"target_type must be one of {TARGET_TYPES}, got {target_type!r}")
        self.n_cls = int(n_cls)
        self.method, self.cl_mode, self.mix_layout = method, cl_mode, mix_layout
        self.gaussian_blur = (method != "ce") if gaussian_blur is None else bool(gaussian_blur)
        self.mix_beta, self.temperature, self.target_type = float(mix_beta), float(temperature), target_type
        self.alpha_time, self.alpha_freq, self.alpha = float(alpha_time), float(alpha_freq), float(alpha)
        self.detach_mix_2d = bool(detach_mix_2d)
        self.input_layout, self.resize_to, self.amp = input_layout, tuple(resize_to), bool(amp)
        self.final_feat_dim = 768

        # ast_models.py:313-317 — VSSM(medium); the blur flag decides the set of blur blocks (without blur the original uses plain vmamba.VSSM = same structure)
        self.v = build_dass_medium_vssm(
            blur=self.gaussian_blur, blur_blocks=tuple(blur_blocks), blur_stage=tuple(blur_stage), kernel_size=kernel_size,
            sigma=sigma, blur_strength=blur_strength, enable_patch_mix=(method == "patchmix_cl"), mix_beta=mix_beta,
            mix_layout=mix_layout, ssm_backend=ssm_backend, drop_path_rate=drop_path_rate,
        )
        # ast_models.py:325-336 — head replaced, patch_embed[0] = Conv2d(1,48,3,2,1) (random init → overwritten by the ckpt)
        self.v.classifier.head = nn.Linear(self.v.classifier.head.in_features, self.n_cls)
        self.v.patch_embed[0] = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.dass_load_info = None
        if audioset_pretrain:
            self.dass_load_info = self.load_dass(pretrained_path)

        # main.py:265, 272
        self.projector = Projector(self.final_feat_dim, proj_dim) if method == "patchmix_cl" else nn.Identity()
        # The original main.py:301 uses `optim_params = list(model.parameters())` and **does not put the projector in the optimizer**
        # — the projector is a separate object from model. The contrastive projection head therefore **stays fixed at its random init**.
        # main_v2.py:279 adds it with `+ list(projector.parameters())`. According to the authors' runs the difference is large:
        #   main.py (not trained) 63.00±0.84 (59 runs)  vs  main_v2.py (trained) 61.39±0.92 (40 runs)  → **−1.61**
        # In this port the projector is a model submodule, so `self.parameters()` includes it automatically (main_v2 behavior).
        # train_projector=False follows main.py (blocks weight updates only; gradients still flow to the backbone).
        self.train_projector = bool(train_projector)
        if not self.train_projector:
            for prm in self.projector.parameters():
                prm.requires_grad_(False)
        self.patchmix_con_loss = PatchMixConLoss(temperature=self.temperature, negative_pair=negative_pair)
        logger.info("dass_lungsrad: method=%s cl_mode=%s mix_layout=%s blur=%s selective_scan=%s(cuda kernel %s)", method, cl_mode, mix_layout,
                    self.gaussian_blur, ssm_backend, cuda_kernel_available())

    # ----------------------------------------------------------------- construction
    def load_dass(self, path: Optional[str]) -> dict:
        from rsc_toolkit.paths import resolve_pretrained

        p = resolve_pretrained(DASS_CKPT_CANDIDATES, explicit=path)  # RSCTK_PRETRAINED_ROOT, then cwd
        if p is None or not p.exists():
            raise FileNotFoundError(f"DASS ckpt not found: {path or DASS_CKPT_CANDIDATES}")
        info = load_dass_checkpoint(self.v, str(p))
        logger.info("Loaded DASS ckpt %s: %d keys, missing %s", p, info["loaded"], info["missing"])
        return info

    # ------------------------------------------------------------------ execution
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """Toolkit fbank (B,128,798) → (B,1,128,798) → bilinear (128,1024) → (B,1,128,1024)  (= original Resize((1024,128)) + unsqueeze/transpose)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout != "freq_time":
            x = x.transpose(-2, -1)
        if tuple(x.shape[-2:]) != self.resize_to:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)
        return x

    def forward_features(self, images: torch.Tensor):
        """(features 768, logits) — original `features, output = model(images)`."""
        return self.v.forward_features(images)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images = self.to_model_input(x)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=self.amp and images.is_cuda):
            _, logits = self.forward_features(images)
        return logits.float()

    def _proj1(self, feats: torch.Tensor) -> torch.Tensor:
        if self.target_type == "grad_block":
            return feats.detach().clone()
        if self.target_type == "grad_flow":
            return feats
        if self.target_type == "project_block":
            return self.projector(feats).detach().clone()
        return self.projector(feats)  # project_flow

    def _axis_cl(self, images, labels, proj1, mix_type: str) -> torch.Tensor:
        mix_feat, _, labels_b, lam, index = self.v.forward_features(images, y=labels, patch_mix=True, mix_type=mix_type)
        mix_feat = mix_feat.detach().clone()          # main.py:326/333 deepcopy(mix.detach()) = Eq.8 sg
        proj_mix = self.projector(mix_feat)
        return self.patchmix_con_loss(proj1, proj_mix, labels, labels_b, lam, index)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """method branches of the original main.py train() (eval mode always takes the CE path, as in the original validate)."""
        images = self.to_model_input(audio_batch)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=self.amp and images.is_cuda):
            feats, logits = self.forward_features(images)
            loss = loss_func(logits, labels)
            if self.training and self.method == "patchmix_cl":
                proj1 = self._proj1(feats)
                if self.cl_mode in ("time", "dual"):
                    loss = loss + self.alpha_time * self._axis_cl(images, labels, proj1, "time")
                if self.cl_mode in ("freq", "dual"):
                    loss = loss + self.alpha_freq * self._axis_cl(images, labels, proj1, "freq")
                if self.cl_mode == "dual_seq":
                    # variant that revives the dead code patch_mix_dual_axis — one forward, one contrastive loss.
                    # It is a **different method** from the default dual (2 independent forwards, 2 losses), so a single alpha coefficient is used.
                    loss = loss + self.alpha * self._axis_cl(images, labels, proj1, "dual_seq")
                if self.cl_mode == "2d":
                    mix_feat, _, labels_b, lam, index = self.v.forward_features(images, y=labels, patch_mix=True, mix_type="2d")
                    if self.detach_mix_2d:
                        mix_feat = mix_feat.detach().clone()
                    proj2 = self.projector(mix_feat)
                    loss = loss + self.alpha * self.patchmix_con_loss(proj1, proj2, labels, labels_b, lam, index)
        return logits.float(), loss
