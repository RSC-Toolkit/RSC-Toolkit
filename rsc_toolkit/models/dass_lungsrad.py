"""
Lung-SRAD (Interspeech 2026, arXiv 2606.11922; RSC-Toolkit/Lung-SRAD d28d783) — DASS-medium(VMamba-S v2) + Spectral-Aware
Regularization(Gaussian blur hook) + Dual-Axis Patch-Mix CL, 공식 코드 충실 이식 (P0).

원본 main.py set_model/train (--model dass, model_size medium 하드코딩):
  images (B,798,128) fbank(per-sample (x−mean)/(2·std)) → [SpecAugment icbhi_ast_sup] → Resize((1024,128)) → (B,1024,128)
  DASS.forward: unsqueeze(1) → transpose(2,3) → (B,1,128,1024) → VSSM(patch_embed → (B,96,32,256) → 4 stage → norm/avgpool 768) → head(768,4)
  method ce      : loss = CE(logits, y)                                   (= "DASS FT";  --gaussian_blur 없음 → blur 블록 ∅)
  method ce_blur : 같은 손실, blur 블록 {2,3}@stage2 활성                 (= "SAR"; 원본 method ce + --gaussian_blur)
  method patchmix_cl (cl_mode dual, 기본): L = CE + α_t·L_PM^time + α_f·L_PM^freq  (main.py:310-351)
      proj1 = projector(features)  (target_type project_flow — 원본 경로엔 gradient 흐름)
      time/freq 각각 독립 forward(독립 index·λ) → mix 특징 detach(=Eq.8 sg) → projector → PatchMixConLoss(τ 0.2, negative_pair all)
      cl_mode time|freq 는 한 축만, cl_mode 2d 는 AST-style(원본은 **detach 없음**, α=alpha) — 원본 코드 그대로(detach_mix_2d=False).
  autocast(fp16) 만, GradScaler 없음 (main.py:305, 364-370; ast_models.py:361 @autocast 이중 적용) → 모델 내부 autocast, trainer precision 32.

사전학습: DASS_medium_v2.pth (Saurabhbhati/DASS v0.2, 'module.v.' 접두어 제거·classifier.head 제외 → head 2키만 missing),
  patch_embed[0] 은 Conv2d(1,48,3,2,1) 로 새로 만들어(imagenet_pretrain=False) ckpt 의 단일채널 가중치로 덮어씀 (ast_models.py:328-336).

quirk (코드 기본값 유지, 플래그로 논문 정의 병행):
  - mix_layout=as_code(기본): patch-mix 가 channel-first (B,96,32,256) 를 (B,H,W,C) 로 오인 → 'time' 은 주파수 32bin, 'freq' 는 채널 96 을 섞고
    λ 분모도 32/96, 시간 축은 안 섞임. as_paper: Eq.6-7(시간 W=256 / 주파수 H=32). (backbone/vmamba/vssm.py 주석 참고)
  - AdamW(코드 misc.py) vs Adam(논문) → 코드.  drop_path 0.3 유지.  blur 는 평가에도 적용되는 고정 연산.
  - SpecAugment 는 **데이터 config(specaugment.enabled, icbhi_ast_sup/mean, randn 게이트)** 에서 datamodule 이 적용한다 — 모델 안에서는 적용하지 않는다(중복 금지).
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
CL_MODES = ("dual", "time", "freq", "2d")
TARGET_TYPES = ("grad_block", "grad_flow", "project_block", "project_flow")


@ModelFactory.register_model_class("dass_lungsrad")
class DASSLungSRAD(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        method: str = "ce",                       # ce (FT) | ce_blur (SAR) | patchmix_cl (Lung-SRAD 및 Table 2 변형)
        cl_mode: str = "dual",                    # patchmix_cl: dual | time | freq | 2d
        mix_layout: str = "as_code",              # as_code (원본 동작) | as_paper (Eq.6-7)
        gaussian_blur: Optional[bool] = None,     # None → method != 'ce' (원본 --gaussian_blur 스크립트 기본)
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
        alpha: float = 1.0,                       # cl_mode 2d 의 계수 (원본 --alpha)
        detach_mix_2d: bool = False,              # 원본 2d 분기는 detach 없음 (quirk); True 면 Eq.8 처럼 sg
        drop_path_rate: float = 0.3,
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,
        input_layout: str = "freq_time",          # 툴킷 fbank (B,128,798)
        resize_to: Tuple[int, int] = (128, 1024), # (F, T) — 원본 Resize((1024,128)) on (T,F)
        amp: bool = True,                         # 원본 autocast fp16, GradScaler 없음 (trainer precision 32)
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

        # ast_models.py:313-317 — VSSM(medium), blur 여부로 blur 블록 집합 결정 (원본은 blur 없으면 plain vmamba.VSSM = 같은 구조)
        self.v = build_dass_medium_vssm(
            blur=self.gaussian_blur, blur_blocks=tuple(blur_blocks), blur_stage=tuple(blur_stage), kernel_size=kernel_size,
            sigma=sigma, blur_strength=blur_strength, enable_patch_mix=(method == "patchmix_cl"), mix_beta=mix_beta,
            mix_layout=mix_layout, ssm_backend=ssm_backend, drop_path_rate=drop_path_rate,
        )
        # ast_models.py:325-336 — head 교체, patch_embed[0] = Conv2d(1,48,3,2,1) (무작위 초기화 → ckpt 로 덮음)
        self.v.classifier.head = nn.Linear(self.v.classifier.head.in_features, self.n_cls)
        self.v.patch_embed[0] = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.dass_load_info = None
        if audioset_pretrain:
            self.dass_load_info = self.load_dass(pretrained_path)

        # main.py:265, 272
        self.projector = Projector(self.final_feat_dim, proj_dim) if method == "patchmix_cl" else nn.Identity()
        self.patchmix_con_loss = PatchMixConLoss(temperature=self.temperature, negative_pair=negative_pair)
        logger.info("dass_lungsrad: method=%s cl_mode=%s mix_layout=%s blur=%s selective_scan=%s(cuda kernel %s)", method, cl_mode, mix_layout,
                    self.gaussian_blur, ssm_backend, cuda_kernel_available())

    # ----------------------------------------------------------------- 구성
    def load_dass(self, path: Optional[str]) -> dict:
        p = pathlib.Path(path) if path else next((pathlib.Path(c) for c in DASS_CKPT_CANDIDATES if pathlib.Path(c).exists()), None)
        if p is None or not p.exists():
            raise FileNotFoundError(f"DASS ckpt 가 없다: {path or DASS_CKPT_CANDIDATES}")
        info = load_dass_checkpoint(self.v, str(p))
        logger.info("DASS ckpt 로드 %s: %d 키, missing %s", p, info["loaded"], info["missing"])
        return info

    # ------------------------------------------------------------------ 실행
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """툴킷 fbank (B,128,798) → (B,1,128,798) → bilinear (128,1024) → (B,1,128,1024)  (= 원본 Resize((1024,128)) + unsqueeze/transpose)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout != "freq_time":
            x = x.transpose(-2, -1)
        if tuple(x.shape[-2:]) != self.resize_to:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)
        return x

    def forward_features(self, images: torch.Tensor):
        """(features 768, logits) — 원본 `features, output = model(images)`."""
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
        """원본 main.py train() 의 method 분기 (평가 모드는 원본 validate 처럼 항상 CE 경로)."""
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
                if self.cl_mode == "2d":
                    mix_feat, _, labels_b, lam, index = self.v.forward_features(images, y=labels, patch_mix=True, mix_type="2d")
                    if self.detach_mix_2d:
                        mix_feat = mix_feat.detach().clone()
                    proj2 = self.projector(mix_feat)
                    loss = loss + self.alpha * self.patchmix_con_loss(proj1, proj2, labels, labels_b, lam, index)
        return logits.float(), loss
