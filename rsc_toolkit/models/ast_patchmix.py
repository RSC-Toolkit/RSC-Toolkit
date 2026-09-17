"""
AST model of the Patch-Mix paper (Interspeech 2023) — faithful port of the official code
(exact port of the official code, verified against goldens).

Sources:
- Paper: Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on
  Respiratory Sound Classification (https://arxiv.org/abs/2305.14032)
- Official code: https://github.com/raymin0223/patch-mix_contrastive_learning
  models/ast.py (MIT, Copyright (c) Sangmin Bae; derived from YuanGongND/ast)
- The same architecture is shared by follow-up repositories such as kaen2891/sgscl (ICASSP 2024) and wa976/PAFA.

This module keeps structure, initialization, and forward identical to the official code.
Behavior of the official code that "looks like a bug" is kept as well. In particular:

**Swapped input-dimension quirk (keep as-is):**
The original main.py:242-243 builds the model for an (h,w)=(798,128) spectrogram with
input_fdim=798, input_tdim=128 — i.e. with frequency/time swapped. As a result the
AudioSet pos_embed (12x101 grid) is cropped to t_dim=12 and interpolated to f_dim=79,
while after the transpose(2,3) in forward the actual conv patch grid is freq 12 x time 79,
so pos_embed is fine-tuned with a layout that does not match the original AST's intent.
The paper's numbers were trained in this state, so it is reproduced here as-is.
(The defaults are the paper's ICBHI setting: input_fdim=798, input_tdim=128)

For a general-purpose "correct" AST use rsc_toolkit.models.ast.ASTClassifier.

Non-structural differences in this toolkit (no effect on weights or numerics):
- Uses the DeiT-distilled skeleton of the toolkit's pinned timm (1.x) instead of timm 0.4.5.
  Parameter key names and shapes were confirmed identical, and golden tests pin this.
- The pretrained ckpt defaults to the archive in pretrained_models/upstream/ (with provenance
  record); if absent it is downloaded from the original Dropbox URL.
- No @autocast decorator on forward. AMP is applied in the training loop
  (the original also wraps train/validate in autocast).
- The original set_model (main.py:259) classifier = deepcopy(model.mlp_head) is
  performed in the constructor and stored as self.classifier. As in the original, mlp_head
  remains unused in forward (it is also in the original optimizer but its parameters
  receive no gradient).
"""
from __future__ import annotations

import logging
import os
import pathlib
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .projector import Projector

try:  # timm is a dependency of the models extra
    import timm
    from timm.models.layers import to_2tuple, trunc_normal_
except Exception as e:  # pragma: no cover
    timm = None
    to_2tuple = None
    trunc_normal_ = None
    _timm_import_error = e

logger = logging.getLogger(__name__)

AUDIOSET_CKPT_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
# archived ckpt candidates (relative to the project root), searched in order.
AUDIOSET_CKPT_CANDIDATES = (
    "pretrained_models/upstream/audioset_0.4593.pth",
    "pretrained_models/audioset_10_10_0.4593.pth",  # file name used by the original code
)


class PatchEmbed(nn.Module):
    """timm PatchEmbed replacement, same as the original — conv patchification without the input-size constraint."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def _create_deit_skeleton(model_size: str, pretrained: bool):
    """Create the timm DeiT-distilled skeleton (0.4.5 name → fall back to the 1.x name)."""
    names = {
        "tiny224": ("vit_deit_tiny_distilled_patch16_224", "deit_tiny_distilled_patch16_224"),
        "small224": ("vit_deit_small_distilled_patch16_224", "deit_small_distilled_patch16_224"),
        "base224": ("vit_deit_base_distilled_patch16_224", "deit_base_distilled_patch16_224"),
        "base384": ("vit_deit_base_distilled_patch16_384", "deit_base_distilled_patch16_384"),
    }
    if model_size not in names:
        raise ValueError("model_size must be one of tiny224, small224, base224, base384")
    old, new = names[model_size]
    try:
        return timm.create_model(old, pretrained=pretrained)
    except (RuntimeError, Exception) as exc:  # timm 1.x does not know the old names
        if "Unknown model" not in str(exc):
            raise
        return timm.create_model(new, pretrained=pretrained)


def _resolve_audioset_ckpt(pretrained_path: Optional[str]) -> pathlib.Path:
    from rsc_toolkit.paths import download_target, resolve_pretrained

    if pretrained_path is not None:
        return resolve_pretrained(explicit=pretrained_path)
    found = resolve_pretrained(AUDIOSET_CKPT_CANDIDATES)  # searched in RSCTK_PRETRAINED_ROOT, then cwd
    if found is not None:
        return found
    # otherwise download from the original source (with a download provenance record)
    from rsc_toolkit.datasets.download_utils import download_file, record_download_provenance

    target = download_target(AUDIOSET_CKPT_CANDIDATES[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("AudioSet AST ckpt not found; downloading: %s", AUDIOSET_CKPT_URL)
    download_file(AUDIOSET_CKPT_URL, target)
    record_download_provenance(target, url=AUDIOSET_CKPT_URL)
    return target


@ModelFactory.register_model_class("ast_patchmix")
class PatchMixAST(BaseRSCModel, nn.Module):
    """
    Paper-faithful model combining the original ASTModel with the classifier setup of set_model.

    forward(x) returns classification logits (x: [B, 1, 798, 128] or [B, 798, 128]).
    The training methods (patchmix / patchmix_cl) use forward_features(..., patch_mix=True).
    """

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        # Note: the two defaults below are swapped on purpose (see the quirk in the module docstring).
        input_fdim: int = 798,
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        mix_beta: float = 1.0,
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        # ---- training method (--method and related arguments of the original main.py) ----
        method: str = "ce",              # "ce" | "patchmix" | "patchmix_cl"
        proj_dim: int = 768,
        temperature: float = 0.06,
        alpha: float = 1.0,
        negative_pair: str = "all",
        target_type: str = "grad_block",
        time_domain: bool = False,
        # Toolkit datamodules provide (B, n_mels, frames), so use "freq_time".
        # For direct (B, frames, n_mels) input as in the original, use "time_freq" (default, faithful).
        input_layout: str = "time_freq",
    ):
        super().__init__()
        if timm is None:  # pragma: no cover
            raise ImportError("PatchMixAST requires timm (`pip install rsc-toolkit[models]`).") from _timm_import_error

        self.final_feat_dim = 768
        self.mix_beta = mix_beta

        if audioset_pretrain and not imagenet_pretrain:
            # same constraint as the original (ast.py:114-115)
            raise ValueError("audioset-only pretraining is not supported; set imagenet_pretrain=True")

        if not audioset_pretrain:
            self._build_imagenet_or_scratch(
                n_cls, fstride, tstride, input_fdim, input_tdim, imagenet_pretrain, model_size, verbose
            )
        else:
            if model_size != "base384":
                raise ValueError("currently only has base384 AudioSet pretrained model.")
            self._build_audioset(n_cls, fstride, tstride, input_fdim, input_tdim, pretrained_path, verbose)

        # original main.py:259 — classifier is a deepcopy of the (newly initialized) mlp_head.
        self.classifier = deepcopy(self.mlp_head)

        if method not in {"ce", "patchmix", "patchmix_cl"}:
            raise ValueError(f"method must be ce|patchmix|patchmix_cl, got {method!r}")
        if input_layout not in {"time_freq", "freq_time"}:
            raise ValueError(f"input_layout must be time_freq|freq_time, got {input_layout!r}")
        self.method = method
        self.alpha = alpha
        self.target_type = target_type
        self.time_domain = time_domain
        self.input_layout = input_layout
        # original set_model: Projector only for patchmix_cl, otherwise Identity (optimizer membership identical too)
        if method == "patchmix_cl":
            self.projector = Projector(self.final_feat_dim, proj_dim)
            from rsc_toolkit.methods.patchmix import PatchMixConLoss

            self.patchmix_con_loss = PatchMixConLoss(temperature=temperature, negative_pair=negative_pair)
        else:
            self.projector = nn.Identity()

    # ------------------------------------------------------------------ construction
    def _build_imagenet_or_scratch(self, n_cls, fstride, tstride, input_fdim, input_tdim,
                                   imagenet_pretrain, model_size, verbose):
        """Same as the original ast.py:56-110 (audioset_pretrain == False branch)."""
        self.v = _create_deit_skeleton(model_size, pretrained=imagenet_pretrain)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)  # keeps even the original typo (oringal)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls)
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self._replace_patch_embed(input_fdim, input_tdim, fstride, tstride, num_patches,
                                  copy_imagenet_proj=imagenet_pretrain)
        if verbose:
            logger.info("PatchMixAST(non-audioset): f_dim=%d t_dim=%d patches=%d", f_dim, t_dim, num_patches)

        if imagenet_pretrain:
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            )
            if t_dim <= self.oringal_hw:
                start = int(self.oringal_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                )
            if f_dim <= self.oringal_hw:
                start = int(self.oringal_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            new_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 2, self.original_embedding_dim)
            )
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)

    def _build_audioset(self, n_cls, fstride, tstride, input_fdim, input_tdim, pretrained_path, verbose):
        """
        Produces the same result as the original ast.py:113-157 (audioset_pretrain == True branch).

        The original builds an inner ASTModel(128,1024, imagenet=False), wraps it in DataParallel,
        loads the full ckpt, and takes only .v. Here the same result is constructed directly:
        (1) skeleton + 1-channel conv patchification, (2) reset pos_embed to the ckpt shape (1, 1214, 768),
        (3) load the ckpt module.v.* weights, (4) crop/interpolate based on the 12x101 grid.
        """
        self.v = _create_deit_skeleton("base384", pretrained=False)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        # number of patches of the inner model (128x1024, fstride/tstride 10) = 12*101 = 1212
        inner_patches = 12 * 101
        self._replace_patch_embed(input_fdim, input_tdim, fstride, tstride, inner_patches,
                                  copy_imagenet_proj=False)
        self.v.pos_embed = nn.Parameter(
            torch.zeros(1, inner_patches + 2, self.original_embedding_dim)
        )

        ckpt_path = _resolve_audioset_ckpt(pretrained_path)
        sd = torch.load(ckpt_path, map_location="cpu")
        v_state = {}
        for key, value in sd.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("v."):
                v_state[key[len("v."):]] = value
            # module.mlp_head.* (527-class AudioSet head) is discarded by the outer model in the original too
        missing, unexpected = self.v.load_state_dict(v_state, strict=False)
        # head/head_dist are unused classification heads that may exist in either the ckpt or the skeleton
        real_missing = [k for k in missing if not k.startswith(("head", "head_dist"))]
        real_unexpected = [k for k in unexpected if not k.startswith(("head", "head_dist"))]
        if real_missing or real_unexpected:
            raise RuntimeError(
                f"AudioSet ckpt load differs from the original. missing={real_missing[:5]} unexpected={real_unexpected[:5]}"
            )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls)
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose:
            logger.info("PatchMixAST(audioset): f_dim=%d t_dim=%d patches=%d", f_dim, t_dim, num_patches)

        # original ast.py:144-157 as-is (including constants 1212/12/101 and crop centers 50/6)
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
        if t_dim < 101:
            new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")
        if f_dim < 12:
            new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2) : 6 - int(f_dim / 2) + f_dim, :]
        elif f_dim > 12:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
        new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def _replace_patch_embed(self, input_fdim, input_tdim, fstride, tstride, num_patches, copy_imagenet_proj):
        """Replace the timm skeleton's patch_embed with the unconstrained PatchEmbed, as in the original."""
        new_patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim), patch_size=16, in_chans=1,
            embed_dim=self.original_embedding_dim,
        )
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if copy_imagenet_proj:
            # original ast.py:83-84 — collapse the RGB 3-channel conv weights to 1 channel by summing over channels
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        new_patch_embed.proj = new_proj
        new_patch_embed.num_patches = num_patches
        self.v.patch_embed = new_patch_embed

    # ------------------------------------------------------------- original methods
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024) -> Tuple[int, int]:
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        return test_out.shape[2], test_out.shape[3]

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        return patch.reshape(B, h, w, dim)

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        return square.reshape(B, h * w, dim)

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        """Same as the original ast.py:178-208. lam is recomputed as the actual replacement ratio."""
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        index = torch.randperm(batch_size).to(device)

        if not time_domain:
            num_mask = int(num_patch * (1.0 - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1.0 - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)

        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    def forward_features(self, x, y=None, patch_mix=False, time_domain=False):
        """Same as the original ASTModel.forward (only the autocast decorator is omitted — see module docstring)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index

    def _to_paper_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Convert toolkit features (freq_time: B,128,798) to the original input (time_freq: B,798,128)."""
        if self.input_layout == "freq_time":
            return x.transpose(-2, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(self._to_paper_layout(x)))

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):  # kwargs: base.py contract — kept open so data configs that attach meta also work
        """
        Same as the method branches of the original main.py train(). Eval mode (validate) always
        takes the ce path, as in the original (original validate is features→classifier→criterion[0]).

        loss_func is produced by build_loss: the ce method uses weighted CE
        (--weighted_loss), patchmix/patchmix_cl use unweighted CE (same as the scripts).
        """
        x = self._to_paper_layout(audio_batch)

        if (not self.training) or self.method == "ce":
            feats = self.forward_features(x)
            logits = self.classifier(feats)
            return logits, loss_func(logits, labels)

        if self.method == "patchmix":
            from rsc_toolkit.methods.patchmix import PatchMixLoss

            mix_feats, y_a, y_b, lam, index = self.forward_features(
                x, y=labels, patch_mix=True, time_domain=self.time_domain
            )
            logits = self.classifier(mix_feats)
            loss = PatchMixLoss(criterion=loss_func)(logits, y_a, y_b, lam)
            return logits, loss

        # patchmix_cl (original main.py:355-372)

        feats = self.forward_features(x)
        logits = self.classifier(feats)
        loss = loss_func(logits, labels)

        if self.target_type == "grad_block":
            proj1 = feats.detach().clone()
        elif self.target_type == "grad_flow":
            proj1 = feats
        elif self.target_type == "project_block":
            proj1 = self.projector(feats).detach().clone()
        elif self.target_type == "project_flow":
            proj1 = self.projector(feats)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type!r}")

        mix_feats, y_a, y_b, lam, index = self.forward_features(
            x, y=labels, patch_mix=True, time_domain=self.time_domain
        )
        proj2 = self.projector(mix_feats)
        loss = loss + self.alpha * self.patchmix_con_loss(proj1, proj2, labels, y_b, lam, index)
        return logits, loss
