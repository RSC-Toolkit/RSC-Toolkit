"""
ADD-RSC (Interspeech 2025, deegy666/ADD-RSC e2b0f82) — adaptive differential denoising layer + AST classifier, faithful port
of the official code (exact port of the official code, verified against goldens).

Original main.py set_model/train (--model ast, defaults):
  images (B,1,798,128) fbank → [SpecAugment] → Resize((1024,256)) bilinear → squeeze(1) → (B,1024,256)
  denoise_images, aux_logits = DiffTransformerLayer(256, 8, depth 6)(images)          # single layer; aux logits = Linear on the mean of the first two frames
  features = ASTModel(fstride=tstride=16, input_fdim=1024, input_tdim=256)(denoise_images.unsqueeze(1))   # (cls+dist)/2, 768
  output = classifier(features)                                                          # deepcopy(mlp_head) = LN + Linear(768, n_cls)
  loss = beta·LabelSmoothing(aux_logits) + (1−beta)·CE(output)    (beta 0.5; --weighted_loss raises NameError in the original and is unusable → unweighted)
  Both train and validate use torch.cuda.amp.autocast (fp16) — no GradScaler.

AST pretrained loading (models/ast.py audioset_pretrain branch): onto a random DeiT-B/384 skeleton (16×16 stride 16, grid 64×16 = 1024 patches,
pos_embed trunc_normal), **only keys with matching name and shape** are copied from the AudioSet 10/10 ckpt (YuanGong
audioset_10_10_0.4593.pth, module.v.* keys) → pos_embed (1214 ≠ 1026) is not loaded (stays random); patch_embed.proj, blocks, norm,
cls/dist are loaded. Then pos_embed[2:] is viewed as (1,768,16,64) and bilinearly interpolated to (f_dim 64, t_dim 16) — because of the
transpose in forward the actual conv grid is 16×64, so the layout gets scrambled, but this is kept as in the original.
The ResNet50 variant (no pretraining) is out of scope.
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
        input_fdim: int = 1024, input_tdim: int = 256,          # original ASTModel arguments (swapped values kept as-is)
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,                  # YuanGong audioset_10_10_0.4593.pth (falls back to the toolkit archive)
        model_size: str = "base384",
        denoise_d_model: int = 256, denoise_num_heads: int = 8, denoise_depth: int = 6,
        loss_beta: float = 0.5,
        label_smoothing: float = 0.20,
        input_layout: str = "freq_time",                        # toolkit fbank (B,128,798) → (B,798,128) → Resize(1024,256)
        resize_to: Tuple[int, int] = (1024, 256),
        amp: bool = True,                                       # original autocast(fp16) — no GradScaler (used with trainer precision 32)
    ):
        super().__init__()
        self.input_layout, self.resize_to, self.loss_beta, self.amp = input_layout, tuple(resize_to), float(loss_beta), bool(amp)
        self.final_feat_dim = 768

        # --- set_model order: denoising layer → ASTModel → classifier ---
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
        # mlp_head of the inner ASTModel(label_dim=527) — the ckpt mlp_head is loaded here but later replaced by a new mlp_head (as in the original)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, 527))
        self.audioset_load_info = None
        if audioset_pretrain:
            self.audioset_load_info = self.load_audioset_matching_keys(pretrained_path)  # load result with the 527-class head
            # ast.py:154-159 — view the random pos_embed as a (16, 64) grid and bilinearly interpolate to (f_dim, t_dim)=(64, 16) (transposed grid, as in the original)
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1024, 768).transpose(1, 2).reshape(1, 768, 16, 64)
            new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls))
        self.classifier = deepcopy(self.mlp_head)  # main.py:262
        self.denoise_criterion = LabelSmoothingLoss(size=n_cls, smoothing=label_smoothing)

    # ----------------------------------------------------------------- construction
    def get_shape(self, fstride, tstride, input_fdim, input_tdim):
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        out = test_proj(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def load_audioset_matching_keys(self, path: Optional[str]) -> dict:
        """ast.py:130-137 — copy only DataParallel keys (module.v.*, module.mlp_head.*) with matching name and shape."""
        p = pathlib.Path(path) if path else next((pathlib.Path(c) for c in AUDIOSET_CKPT_CANDIDATES if pathlib.Path(c).exists()), None)
        if p is None or not p.exists():
            raise FileNotFoundError(f"AudioSet ckpt not found: {path or AUDIOSET_CKPT_CANDIDATES}")
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
        logger.info("ADD-RSC AudioSet load: %d keys, shape mismatch (not loaded) %s", len(loaded), mismatch)
        return {"loaded": loaded, "mismatch": mismatch}

    # ------------------------------------------------------------------ execution
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """Toolkit fbank (B,128,798) → (B,798,128) → Resize((1024,256)) bilinear (upsampling, so antialias has no effect) → (B,1024,256)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)
        x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)
        return x.squeeze(1)

    def forward_ast(self, x: torch.Tensor) -> torch.Tensor:
        """ast.py forward: (B,1,1024,256) → transpose(2,3) → patches → cls/dist → blocks → (cls+dist)/2."""
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
                loss = denoise_loss + class_loss  # validate(): unweighted sum (for logging)
        return output.float(), loss
