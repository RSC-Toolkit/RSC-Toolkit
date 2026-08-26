"""
ADD-RSC(Interspeech 2025, deegy666/ADD-RSC e2b0f82) — 적응 차분 잡음제거 층 + AST 분류기, 공식 코드 충실 이식 (P0).

원본 main.py set_model/train (--model ast, 기본값):
  images (B,1,798,128) fbank → [SpecAugment] → Resize((1024,256)) bilinear → squeeze(1) → (B,1024,256)
  denoise_images, aux_logits = DiffTransformerLayer(256, 8, depth 6)(images)          # 단일 층, 보조 로짓 = 앞 두 프레임 평균의 Linear
  features = ASTModel(fstride=tstride=16, input_fdim=1024, input_tdim=256)(denoise_images.unsqueeze(1))   # (cls+dist)/2, 768
  output = classifier(features)                                                          # deepcopy(mlp_head) = LN + Linear(768, n_cls)
  loss = beta·LabelSmoothing(aux_logits) + (1−beta)·CE(output)    (beta 0.5; --weighted_loss 는 원본에서 NameError 로 사용 불가 → 비가중)
  train/validate 모두 torch.cuda.amp.autocast (fp16) — GradScaler 없음.

AST 사전학습 로드(models/ast.py audioset_pretrain 분기): 랜덤 DeiT-B/384 골격(16×16 stride 16, 격자 64×16 = 1024 패치, pos_embed trunc_normal)
에 AudioSet 10/10 ckpt(YuanGong audioset_10_10_0.4593.pth, module.v.* 키)에서 **이름·모양이 같은 키만** 복사 → pos_embed(1214 ≠ 1026) 는
미로드(난수 유지), patch_embed.proj·블록·norm·cls/dist 로드. 이어 pos_embed[2:] 를 (1,768,16,64) 로 본 뒤 (f_dim 64, t_dim 16) 로
bilinear 보간 — forward 의 transpose 로 실제 conv 격자는 16×64 라 배열이 뒤섞이지만 원본 그대로 둔다.
ResNet50 변형(사전학습 없음)은 범위 밖.
"""
from __future__ import annotations

import logging
import pathlib
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .addrsc import DiffTransformerLayer, LabelSmoothingLoss
from .ast_patchmix import AUDIOSET_CKPT_CANDIDATES, PatchEmbed, _create_deit_skeleton
from .base import BaseRSCModel
from .factory import ModelFactory

try:
    from timm.models.layers import trunc_normal_
except Exception:  # pragma: no cover
    from timm.layers import trunc_normal_

logger = logging.getLogger(__name__)


@ModelFactory.register_model_class("ast_addrsc")
class ADDRSC(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 16, tstride: int = 16,
        input_fdim: int = 1024, input_tdim: int = 256,          # 원본 ASTModel 인자 (뒤집힌 값 그대로)
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,                  # YuanGong audioset_10_10_0.4593.pth (없으면 툴킷 아카이브)
        model_size: str = "base384",
        denoise_d_model: int = 256, denoise_num_heads: int = 8, denoise_depth: int = 6,
        loss_beta: float = 0.5,
        label_smoothing: float = 0.20,
        input_layout: str = "freq_time",                        # 툴킷 fbank (B,128,798) → (B,798,128) → Resize(1024,256)
        resize_to: Tuple[int, int] = (1024, 256),
        amp: bool = True,                                       # 원본 autocast(fp16) — GradScaler 없음 (trainer precision 32 와 함께 쓴다)
    ):
        super().__init__()
        self.input_layout, self.resize_to, self.loss_beta, self.amp = input_layout, tuple(resize_to), float(loss_beta), bool(amp)
        self.final_feat_dim = 768

        # --- set_model 순서: 잡음제거 층 → ASTModel → classifier ---
        self.bias_denoise_encoder = DiffTransformerLayer(d_model=denoise_d_model, num_heads=denoise_num_heads, depth=denoise_depth, n_cls=n_cls)

        self.v = _create_deit_skeleton(model_size, pretrained=False)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        patch_embed = PatchEmbed(img_size=(input_fdim, input_tdim), patch_size=16, in_chans=1, embed_dim=self.original_embedding_dim)
        patch_embed.proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        patch_embed.num_patches = num_patches
        self.v.patch_embed = patch_embed
        self.v.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.original_embedding_dim))
        trunc_normal_(self.v.pos_embed, std=0.02)
        self.grid = (f_dim, t_dim)
        # 내부 ASTModel(label_dim=527) 의 mlp_head — ckpt 의 mlp_head 가 여기로 로드되지만 이후 새 mlp_head 로 대체된다 (원본 그대로)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, 527))
        self.audioset_load_info = None
        if audioset_pretrain:
            self.audioset_load_info = self.load_audioset_matching_keys(pretrained_path)  # 527 헤드 상태에서의 로드 결과
            # ast.py:154-159 — 난수 pos_embed 를 (16, 64) 격자로 보고 (f_dim, t_dim)=(64, 16) 으로 bilinear 보간 (전치된 격자, 원본 그대로)
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1024, 768).transpose(1, 2).reshape(1, 768, 16, 64)
            new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls))
        self.classifier = deepcopy(self.mlp_head)  # main.py:262
        self.denoise_criterion = LabelSmoothingLoss(size=n_cls, smoothing=label_smoothing)

    # ----------------------------------------------------------------- 구성
    def get_shape(self, fstride, tstride, input_fdim, input_tdim):
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        out = test_proj(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def load_audioset_matching_keys(self, path: Optional[str]) -> dict:
        """ast.py:130-137 — DataParallel 키(module.v.*, module.mlp_head.*) 중 이름·모양이 같은 것만 복사."""
        p = pathlib.Path(path) if path else next((pathlib.Path(c) for c in AUDIOSET_CKPT_CANDIDATES if pathlib.Path(c).exists()), None)
        if p is None or not p.exists():
            raise FileNotFoundError(f"AudioSet ckpt 가 없다: {path or AUDIOSET_CKPT_CANDIDATES}")
        weights = torch.load(p, map_location="cpu", weights_only=True)
        current_v, current_head = self.v.state_dict(), self.mlp_head.state_dict()
        loaded, mismatch = [], []
        for prefix, current in (("module.v.", current_v), ("module.mlp_head.", current_head)):
            for k1, v1 in current.items():
                k2 = prefix + k1
                if k2 in weights:
                    if v1.shape == weights[k2].shape:
                        current[k1] = weights[k2]
                        loaded.append(k2)
                    else:
                        mismatch.append(k2)
        self.v.load_state_dict(current_v, strict=False)
        self.mlp_head.load_state_dict(current_head, strict=False)
        logger.info("ADD-RSC AudioSet 로드: %d 키, 모양 불일치(미로드) %s", len(loaded), mismatch)
        return {"loaded": loaded, "mismatch": mismatch}

    # ------------------------------------------------------------------ 실행
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """툴킷 fbank (B,128,798) → (B,798,128) → Resize((1024,256)) bilinear (업샘플이라 antialias 무관) → (B,1024,256)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)
        x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)
        return x.squeeze(1)

    def forward_ast(self, x: torch.Tensor) -> torch.Tensor:
        """ast.py forward: (B,1,1024,256) → transpose(2,3) → 패치 → cls/dist → blocks → (cls+dist)/2."""
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        x = torch.cat((self.v.cls_token.expand(B, -1, -1), self.v.dist_token.expand(B, -1, -1), x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return (x[:, 0] + x[:, 1]) / 2

    def forward_all(self, images: torch.Tensor):
        """images (B,1024,256) → (output logits, aux denoise logits)."""
        denoise_images, denoise_feature = self.bias_denoise_encoder(images)
        features = self.forward_ast(denoise_images.unsqueeze(1))
        return self.classifier(features), denoise_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(self.to_model_input(x))[0]

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        images = self.to_model_input(audio_batch)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=self.amp and images.is_cuda):
            output, denoise_feature = self.forward_all(images)
            denoise_loss = self.denoise_criterion(denoise_feature, labels)
            class_loss = loss_func(output, labels)
            if self.training:
                loss = self.loss_beta * denoise_loss + (1 - self.loss_beta) * class_loss
            else:
                loss = denoise_loss + class_loss  # validate(): 가중 없이 합 (기록용)
        return output.float(), loss
