# Portions of this file are adapted from M2D (https://github.com/nttcslab/m2d); that code is under the NTT Software License Agreement for Evaluation.
# The sin-cos position embedding follows MAE (https://github.com/facebookresearch/mae), which is under the CC-BY-NC-4.0 license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
M2D encoder (ViT-Base, audio log-mel patches) — re-implements the same computation as nttcslab/m2d `M2DEncoderViT` on top of timm ViT.

References (behavior reproduced, code not copied): nttcslab/m2d m2d/models_mae.py M2DEncoderViT·PatchEmbed·forward_encoder,
m2d/pos_embed.py get_2d_sincos_pos_embed, m2d/runtime_audio.py get_backbone/encode_lms.

Behavior:
- Input (B, 1, 80, T): Conv2d(1, 768, kernel=stride=(16, 4)) patchify (grid 5 x T/4, f-major flattening)
- Fixed 2D sin-cos pos_embed (grid 5 x 50 = 250 + cls) — for short inputs only the leading part of the time axis is sliced (adjust_short)
- cls_token + pos_embed[:, :1] prepended, then timm Block x12 (768, heads 12, mlp 4, qkv_bias, LayerNorm eps 1e-6) → norm
- encode_lms: zero-pad T to a multiple of the patch size, forward in 200-frame chunks, drop cls,
  concat the 5 frequency patches with 'b (f t) d -> b t (f d)' (3840), concat chunks along time
- norm_stats buffer (ckpt value preserved; unused in the TASLP ICBHI path)
Checkpoint: only keys of ckpt['model'] that match this module's parameter/buffer names are loaded (decoder_*, target_* etc. are dropped).
"""
from __future__ import annotations

import logging
import pathlib
from functools import partial
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- sin-cos pos embed (MAE formula)
def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Sequence[int], cls_token: bool = False) -> np.ndarray:
    gH, gW = int(grid_size[0]), int(grid_size[1])
    grid = np.meshgrid(np.arange(gW, dtype=np.float32), np.arange(gH, dtype=np.float32))  # w first (MAE convention)
    grid = np.stack(grid, axis=0).reshape([2, 1, gH, gW])
    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """Conv patchify without a size assert (equivalent to timm 0.4.12 PatchEmbed)."""

    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B, N(f-major), C


class M2DEncoder(nn.Module):
    def __init__(self, img_size=(80, 200), patch_size=(16, 4), in_chans=1, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, norm_stats=(-7.1, 4.2)):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.norm_stats = nn.Parameter(torch.tensor(list(norm_stats), dtype=torch.float32), requires_grad=False)
        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size(), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        nn.init.normal_(self.cls_token, std=0.02)
        self.embed_dim = embed_dim

    def grid_size(self) -> Tuple[int, int]:
        return self.patch_embed.grid_size

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 80, T≤200) → (B, 1 + 5*T/4, D). For short inputs the pos_embed time axis is truncated from the front."""
        x = self.patch_embed(x)
        pos_embed = self.pos_embed[:, 1:, :]
        if x.shape[1] < pos_embed.shape[1]:
            dims = pos_embed.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos_embed = pos_embed.reshape(1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins * frames, dims)
        x = x + pos_embed
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def encode_lms(self, x: torch.Tensor) -> torch.Tensor:
        """runtime_audio.encode_lms (flat_features=False): (B, 1, F, T) → (B, T', 5*D)."""
        fbins = self.grid_size()[0]
        unit_frames = self.patch_embed.img_size[1]
        patch_frames = self.patch_embed.patch_size[1]
        chunks = (x.shape[-1] + unit_frames - 1) // unit_frames
        pad_frames = (patch_frames - (x.shape[-1] % unit_frames % patch_frames)) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))
        embeddings = []
        for i in range(chunks):
            emb = self.forward_encoder(x[..., i * unit_frames:(i + 1) * unit_frames])[:, 1:, :]  # drop cls
            B, N, D = emb.shape
            emb = emb.reshape(B, fbins, N // fbins, D).permute(0, 2, 1, 3).reshape(B, N // fbins, fbins * D)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=1)


def load_m2d_checkpoint(encoder: M2DEncoder, path) -> dict:
    """Strictly loads only the keys of ckpt['model'] that match encoder parameter/buffer names (runtime_audio.drop_non_model_weights)."""
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in state.items()}
    names = {n for n, _ in encoder.named_parameters()} | {n for n, _ in encoder.named_buffers()}
    kept = {k: v for k, v in state.items() if k in names}
    dropped = sorted(k for k in state if k not in names)
    if "norm_stats" not in kept:
        # older ckpts (e.g. 230529) have no norm_stats — runtime_audio fills in the default [-7.1, 4.2]
        kept["norm_stats"] = encoder.norm_stats.detach().clone()
    missing = sorted(names - set(kept))
    if missing:
        raise RuntimeError(f"M2D ckpt is missing encoder keys: {missing[:5]}")
    encoder.load_state_dict(kept, strict=True)
    logger.info("Loaded M2D ckpt %s: used %d keys, dropped %d keys (e.g. %s)", path, len(kept), len(dropped), dropped[:3])
    return {"used": sorted(kept), "dropped": dropped, "args": ckpt.get("args")}
