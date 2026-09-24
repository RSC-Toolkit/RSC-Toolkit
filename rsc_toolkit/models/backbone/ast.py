# Portions of this file are adapted from AST (https://github.com/YuanGongND/ast); that code is under the BSD-3-Clause license.
# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from typing import Tuple
import logging
import os

import torch
import torch.nn as nn

from rsc_toolkit.datasets.download_utils import download_file
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

    AUDIOSET_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"

    def __init__(
        self,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        pretrained: str = "imagenet",
        pretrained_path: str | None = None,
        download_if_missing: bool = True,
        model_size: str = "base384",
        verbose: bool = True,
    ):
        super().__init__()

        if timm is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "ASTBackbone requires timm. Install with `pip install timm`."
            ) from _timm_import_error

        pretrain_source = self._resolve_pretrain_source(pretrained)
        if pretrain_source == "audioset_imagenet" and model_size != "base384":
            raise ValueError("audioset_imagenet weights are only available for model_size=base384")

        if verbose:
            logger.info(
                "AST init: pretrained=%s model_size=%s",
                pretrain_source,
                model_size,
            )

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if model_size == "tiny224":
            model_name = "vit_deit_tiny_distilled_patch16_224"
            fallback_name = "deit_tiny_distilled_patch16_224"
        elif model_size == "small224":
            model_name = "vit_deit_small_distilled_patch16_224"
            fallback_name = "deit_small_distilled_patch16_224"
        elif model_size == "base224":
            model_name = "vit_deit_base_distilled_patch16_224"
            fallback_name = "deit_base_distilled_patch16_224"
        elif model_size == "base384":
            model_name = "vit_deit_base_distilled_patch16_384"
            fallback_name = "deit_base_distilled_patch16_384"
        else:
            raise ValueError("model_size must be one of tiny224, small224, base224, base384")

        try:
            self.v = timm.create_model(model_name, pretrained=pretrain_source == "imagenet")
        except RuntimeError as exc:
            if "Unknown model" not in str(exc):
                raise
            self.v = timm.create_model(fallback_name, pretrained=pretrain_source == "imagenet")

        # Ensure timm's PatchEmbed size assertion matches our spectrogram dims.
        self.v.patch_embed.img_size = (input_fdim, input_tdim)

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
        if pretrain_source == "imagenet":
            new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        if pretrain_source == "imagenet":
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
        elif pretrain_source == "none":
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)
        else:
            self._load_audioset_pretrained(pretrained_path, download_if_missing)
            self._adapt_audioset_pos_embed(f_dim, t_dim)

    def _get_shape(self, fstride: int, tstride: int, input_fdim: int, input_tdim: int) -> Tuple[int, int]:
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def _resolve_pretrain_source(self, pretrained: str) -> str:
        pretrained = pretrained.lower()
        if pretrained not in {"imagenet", "audioset_imagenet", "none"}:
            raise ValueError("pretrained must be one of: imagenet, audioset_imagenet, none")
        return pretrained

    def _load_audioset_pretrained(self, pretrained_path: str | None, download_if_missing: bool) -> None:
        if pretrained_path is None:
            default_dir = os.path.join(os.getcwd(), "pretrained_models")
            pretrained_path = os.path.join(default_dir, "audioset_10_10_0.4593.pth")
        if not os.path.exists(pretrained_path):
            if not download_if_missing:
                raise FileNotFoundError(
                    f"audioset checkpoint not found at {pretrained_path}. "
                    "Download it or pass pretrained_path."
                )
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            download_file(self.AUDIOSET_URL, pretrained_path)
        try:
            state = torch.load(pretrained_path, map_location="cpu")
        except RuntimeError as exc:
            if not "failed reading zip archive" in str(exc):
                raise
            logger.error("Bad archive, re-downloading weights.")
            download_file(self.AUDIOSET_URL, pretrained_path)
            state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        v_state = self.v.state_dict()
        filtered = {}
        audioset_pos_embed = None
        for key, value in state.items():
            normalized = key
            if normalized.startswith("module."):
                normalized = normalized[len("module."):]
            if normalized.startswith("v."):
                normalized = normalized[len("v."):]
            if normalized == "pos_embed":
                audioset_pos_embed = value
                continue
            if normalized in v_state:
                filtered[normalized] = value
        self.v.load_state_dict(filtered, strict=False)
        if audioset_pos_embed is None:
            raise ValueError("audioset checkpoint missing pos_embed")
        self._audioset_pos_embed = audioset_pos_embed

    def _adapt_audioset_pos_embed(self, f_dim: int, t_dim: int) -> None:
        if not hasattr(self, "_audioset_pos_embed"):
            raise ValueError("audioset pos_embed not loaded")
        pos_embed = self._audioset_pos_embed
        audio_tokens = pos_embed.shape[1] - 2
        orig_f_dim, orig_t_dim = 12, 101
        if audio_tokens != orig_f_dim * orig_t_dim:
            raise ValueError(
                f"Unexpected audioset positional tokens: {audio_tokens}. "
                "Update _adapt_audioset_pos_embed for this checkpoint."
            )
        embed_dim = pos_embed.shape[2]
        new_pos_embed = (
            pos_embed[:, 2:, :]
            .detach()
            .reshape(1, orig_f_dim * orig_t_dim, embed_dim)
            .transpose(1, 2)
            .reshape(1, embed_dim, orig_f_dim, orig_t_dim)
        )
        if t_dim < orig_t_dim:
            start = int(orig_t_dim / 2) - int(t_dim / 2)
            new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(orig_f_dim, t_dim), mode="bilinear")
        if f_dim < orig_f_dim:
            start = int(orig_f_dim / 2) - int(f_dim / 2)
            new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
        elif f_dim > orig_f_dim:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
        new_pos_embed = new_pos_embed.reshape(1, embed_dim, f_dim * t_dim).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(torch.cat([pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

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
