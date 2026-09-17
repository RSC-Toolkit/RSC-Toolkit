# Portions of this file are ported from MVST (https://github.com/wentaoheunnc/MVST), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Per-view AST of MVST (ICASSP 2024) — faithful port of the official code (exact port of the official code, verified against goldens).

Sources:
- Paper: Multi-View Spectrogram Transformer for Respiratory Sound Classification (https://arxiv.org/abs/2311.09655)
- Official code: https://github.com/wentaoheunnc/MVST {16,32,64,128,256}/models/ast.py (borrowed from patch-mix ast.py; only the per-view kernel differs)

Architecture (as in the original):
- DeiT-B distilled skeleton. Patch conv kernel = stride = view kernel (16,16)/(32,8)/(64,4)/(128,2)/(256,1);
  input (B,1,1024,256) → transpose(2,3) in forward → (B,1,256,1024) = (freq 256, time 1024) → 1024 patches for every view, pos_embed (1,1026,768).
- audioset_pretrain branch: builds an inner ASTModel(imagenet_pretrain=False, view kernel), then copies from `audioset_16_16_0.4422.pth`
  **only keys whose name and shape both match** (on shape mismatch it prints and leaves the key). As a result pos_embed (514 vs 1026) is not loaded
  for any view (trunc_normal random), patch_embed.proj (768,1,16,16) is loaded only for the 16×16 view, and blocks, norm, cls/dist are loaded for all views.
  The subsequent pos_embed interpolation is to the same grid, i.e. identity.
- Features = (cls + dist)/2, classifier = deepcopy(mlp_head) = LayerNorm(768) + Linear(768, n_cls). mlp_head is unused in forward.
Input pipeline: fbank (798×128, AST constant normalization) → [per-sample SpecAugment] → Resize((1024, 256)) bilinear — the Resize is done by this model.
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
    "pretrained_models/audioset_16_16_0.4422.pth",  # location used by the original code
)
VIEW_KERNELS = {16: (16, 16), 32: (32, 8), 64: (64, 4), 128: (128, 2), 256: (256, 1)}


@ModelFactory.register_model_class("ast_mvst")
class MVSTViewAST(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        view: int = 16,                      # 16 | 32 | 64 | 128 | 256 (kernel (kf, kt) = VIEW_KERNELS[view])
        input_fdim: int = 256,
        input_tdim: int = 1024,
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,
        model_size: str = "base384",
        input_layout: str = "freq_time",     # toolkit fbank (B,128,798) → the model applies (B,798,128)→Resize(1024,256)
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

        # same as the inner ASTModel(imagenet_pretrain=False, audioset_pretrain=False): random skeleton + view kernel + random pos_embed
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
            # original ast.py:154-159 — bilinear interpolation to its own grid (identity)
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, num_patches, 768).transpose(1, 2).reshape(1, 768, f_dim, t_dim)
            new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls))
        self.classifier = deepcopy(self.mlp_head)  # main.py:257

    # ----------------------------------------------------------------- construction
    def get_shape(self, input_fdim, input_tdim):
        test_proj = nn.Conv2d(1, 768, kernel_size=self.kernel, stride=self.kernel)
        out = test_proj(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def load_audioset_matching_keys(self, path: Optional[str]) -> dict:
        """Original ast.py:130-137: copy only keys with matching name and shape (module.v.* → v.*). Returns a summary of the result."""
        from rsc_toolkit.paths import resolve_pretrained

        p = resolve_pretrained(AUDIOSET_16_16_CANDIDATES, explicit=path)  # searched in RSCTK_PRETRAINED_ROOT, then cwd
        if p is None or not p.exists():
            raise FileNotFoundError(f"audioset_16_16_0.4422.pth not found (create it with scripts/reproduction/convert_hf_ast_to_yuangong.py): {AUDIOSET_16_16_CANDIDATES}")
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
        logger.info("MVST view %d AudioSet load: %d keys, shape mismatch (not loaded) %s", self.view, len(loaded), mismatch)
        return {"loaded": loaded, "mismatch": mismatch}

    def load_reference_checkpoint(self, path) -> None:
        """Strict load of an MVST save_model output (best.pth: model=module.v.*|v.* + mlp_head.*, classifier=0/1.*)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {}
        for k, v in ckpt["model"].items():
            k = k[len("module."):] if k.startswith("module.") else k
            k = k[len("backbone."):] if k.startswith("backbone.") else k
            state[k] = v
        state.update({f"classifier.{k}": v for k, v in ckpt["classifier"].items()})
        self.load_state_dict(state, strict=True)
        logger.info("MVST view %d authors' checkpoint loaded: %s (epoch %s)", self.view, path, ckpt.get("epoch"))

    # ------------------------------------------------------------------ execution
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """Toolkit fbank (B,128,798) or (B,798,128) → (B,1,1024,256) (original transforms.Resize, bilinear)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)  # (B,1,798,128) = (time, freq)
        return F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False, antialias=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Original forward: (B,1,1024,256) → transpose(2,3) → patches → cls/dist → blocks → (cls+dist)/2."""

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
