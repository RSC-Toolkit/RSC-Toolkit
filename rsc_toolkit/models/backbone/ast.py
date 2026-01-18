from __future__ import annotations

from typing import Tuple
import logging

import torch
import torch.nn as nn

try: # TODO cleanup
    import timm
    from timm.models.layers import to_2tuple, trunc_normal_
except Exception as e:  # pragma: no cover - optional dependency
    timm = None
    to_2tuple = None
    trunc_normal_ = None
    _timm_import_error = e

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTBackbone(nn.Module):
    """
    Audio Spectrogram Transformer backbone (feature extractor).
    """

    def __init__(
        self,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = True,
        model_size: str = "base384",
        verbose: bool = True,
    ):
        super().__init__()

        if timm is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "ASTBackbone requires timm. Install with `pip install timm`."
            ) from _timm_import_error

        if verbose:
            logger.info(
                "AST init: imagenet_pretrain=%s model_size=%s",
                imagenet_pretrain,
                model_size,
            )

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if model_size == "tiny224":
            self.v = timm.create_model("vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain)
        elif model_size == "small224":
            self.v = timm.create_model("vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain)
        elif model_size == "base224":
            self.v = timm.create_model("vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain)
        elif model_size == "base384":
            self.v = timm.create_model("vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain)
        else:
            raise ValueError("model_size must be one of tiny224, small224, base224, base384")

        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.final_feat_dim = self.original_embedding_dim

        f_dim, t_dim = self._get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        if verbose:
            logger.info("AST patch layout fstride=%d tstride=%d patches=%d", fstride, tstride, num_patches)

        new_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        if imagenet_pretrain:
            new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        if imagenet_pretrain:
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
            )
            if t_dim <= self.original_hw:
                start = int(self.original_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.original_hw, t_dim), mode="bilinear"
                )
            if f_dim <= self.original_hw:
                start = int(self.original_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)

    def _get_shape(self, fstride: int, tstride: int, input_fdim: int, input_tdim: int) -> Tuple[int, int]:
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected [B, F, T] or [B, 1, F, T], got {tuple(x.shape)}")

        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(x.shape[0], -1, -1)
        dist_token = self.v.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return x
