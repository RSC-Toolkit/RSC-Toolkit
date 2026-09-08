"""
MVST(ICASSP 2024) 의 뷰별 AST — 공식 코드 충실 이식 (P0).

출처:
- 논문: Multi-View Spectrogram Transformer for Respiratory Sound Classification (https://arxiv.org/abs/2311.09655)
- 공식 코드: https://github.com/wentaoheunnc/MVST {16,32,64,128,256}/models/ast.py (patch-mix ast.py 차용, 뷰별 커널만 다름)

구조 (원본 그대로):
- DeiT-B distilled 골격. 패치 conv 의 kernel = stride = 뷰 커널 (16,16)/(32,8)/(64,4)/(128,2)/(256,1);
  입력 (B,1,1024,256) → forward 에서 transpose(2,3) → (B,1,256,1024) = (freq 256, time 1024) → 모든 뷰 1024 패치, pos_embed (1,1026,768).
- audioset_pretrain 분기: 내부 ASTModel(imagenet_pretrain=False, 뷰 커널) 을 만든 뒤 `audioset_16_16_0.4422.pth` 에서
  **이름·모양이 모두 같은 키만** 복사 (모양 다르면 print 후 방치). 그 결과 pos_embed(514 vs 1026)는 전 뷰에서 미로드(trunc_normal 난수),
  patch_embed.proj(768,1,16,16)는 16×16 뷰만 로드, 블록·norm·cls/dist 는 전 뷰 로드. 이어지는 pos_embed 보간은 같은 격자라 항등.
- 특징 = (cls + dist)/2, 분류기 = deepcopy(mlp_head) = LayerNorm(768) + Linear(768, n_cls). mlp_head 는 forward 미사용.
입력 파이프라인: fbank(798×128, AST 상수 정규화) → [SpecAugment 샘플별] → Resize((1024, 256)) bilinear — Resize 는 이 모델이 수행한다.
"""
from __future__ import annotations

import logging
import pathlib
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ast_patchmix import PatchEmbed, _create_deit_skeleton
from .base import BaseRSCModel
from .factory import ModelFactory

try:
    from timm.models.layers import trunc_normal_
except Exception:  # pragma: no cover
    from timm.layers import trunc_normal_

logger = logging.getLogger(__name__)

AUDIOSET_16_16_CANDIDATES = (
    "pretrained_models/upstream/audioset_16_16_0.4422.pth",
    "pretrained_models/audioset_16_16_0.4422.pth",  # 원본 코드가 쓰는 위치
)
VIEW_KERNELS = {16: (16, 16), 32: (32, 8), 64: (64, 4), 128: (128, 2), 256: (256, 1)}


@ModelFactory.register_model_class("ast_mvst")
class MVSTViewAST(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        view: int = 16,                      # 16 | 32 | 64 | 128 | 256 (커널 (kf, kt) = VIEW_KERNELS[view])
        input_fdim: int = 256,
        input_tdim: int = 1024,
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,
        model_size: str = "base384",
        input_layout: str = "freq_time",     # 툴킷 fbank (B,128,798) → 모델이 (B,798,128)→Resize(1024,256)
        resize_to: Tuple[int, int] = (1024, 256),
    ):
        super().__init__()
        if view not in VIEW_KERNELS:
            raise ValueError(f"view must be one of {sorted(VIEW_KERNELS)}, got {view}")
        self.view = view
        self.kernel = VIEW_KERNELS[view]
        self.input_layout = input_layout
        self.resize_to = tuple(resize_to)
        self.final_feat_dim = 768

        # 내부 ASTModel(imagenet_pretrain=False, audioset_pretrain=False) 과 동일: 랜덤 골격 + 뷰 커널 + 난수 pos_embed
        self.v = _create_deit_skeleton(model_size, pretrained=False)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        f_dim, t_dim = self.get_shape(input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        patch_embed = PatchEmbed(img_size=(input_fdim, input_tdim), patch_size=self.kernel, in_chans=1,
                                 embed_dim=self.original_embedding_dim)
        patch_embed.proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=self.kernel, stride=self.kernel)
        patch_embed.num_patches = num_patches
        self.v.patch_embed = patch_embed
        self.v.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.original_embedding_dim))
        trunc_normal_(self.v.pos_embed, std=0.02)
        self.grid = (f_dim, t_dim)

        if audioset_pretrain:
            self.load_audioset_matching_keys(pretrained_path)
            # 원본 ast.py:154-159 — 자기 격자로 bilinear 보간 (항등)
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, num_patches, 768).transpose(1, 2).reshape(1, 768, f_dim, t_dim)
            new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls))
        self.classifier = deepcopy(self.mlp_head)  # main.py:257

    # ----------------------------------------------------------------- 구성
    def get_shape(self, input_fdim, input_tdim):
        test_proj = nn.Conv2d(1, 768, kernel_size=self.kernel, stride=self.kernel)
        out = test_proj(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def load_audioset_matching_keys(self, path: Optional[str]) -> dict:
        """원본 ast.py:130-137: 이름·모양이 같은 키만 복사 (module.v.* → v.*). 결과 요약을 돌려준다."""
        from rsc_toolkit.paths import resolve_pretrained

        p = resolve_pretrained(AUDIOSET_16_16_CANDIDATES, explicit=path)  # RSCTK_PRETRAINED_ROOT → cwd 순
        if p is None or not p.exists():
            raise FileNotFoundError(f"audioset_16_16_0.4422.pth 가 없다 (scripts/reproduction/convert_hf_ast_to_yuangong.py 로 생성): {AUDIOSET_16_16_CANDIDATES}")
        weights = torch.load(p, map_location="cpu", weights_only=True)
        current = self.v.state_dict()
        loaded, mismatch = [], []
        for k1, v1 in current.items():
            k2 = "module.v." + k1
            if k2 in weights:
                if v1.shape == weights[k2].shape:
                    current[k1] = weights[k2]
                    loaded.append(k1)
                else:
                    mismatch.append(k1)
        self.v.load_state_dict(current, strict=False)
        logger.info("MVST view %d AudioSet 로드: %d 키, 모양 불일치(미로드) %s", self.view, len(loaded), mismatch)
        return {"loaded": loaded, "mismatch": mismatch}

    def load_reference_checkpoint(self, path) -> None:
        """MVST save_model 산출물(best.pth: model=module.v.*|v.* + mlp_head.*, classifier=0/1.*) strict 로드."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {}
        for k, v in ckpt["model"].items():
            k = k[len("module."):] if k.startswith("module.") else k
            k = k[len("backbone."):] if k.startswith("backbone.") else k
            state[k] = v
        state.update({f"classifier.{k}": v for k, v in ckpt["classifier"].items()})
        self.load_state_dict(state, strict=True)
        logger.info("MVST view %d 저자 체크포인트 로드: %s (epoch %s)", self.view, path, ckpt.get("epoch"))

    # ------------------------------------------------------------------ 실행
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """툴킷 fbank (B,128,798) 또는 (B,798,128) → (B,1,1024,256) (원본 transforms.Resize, bilinear)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)  # (B,1,798,128) = (time, freq)
        return F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """원본 forward: (B,1,1024,256) → transpose(2,3) → 패치 → cls/dist → blocks → (cls+dist)/2."""
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return (x[:, 0] + x[:, 1]) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(self.to_model_input(x)))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
