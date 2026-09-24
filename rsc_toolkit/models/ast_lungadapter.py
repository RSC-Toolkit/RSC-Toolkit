"""
LungAdapter (Interspeech 2024, Xiao et al.) — frozen AST + Attention-Adapter/Res-Adapter, **re-implementation from the paper description**
(no official or unofficial code exists).

Paper: LungAdapter: Efficient Adapting Audio Spectrogram Transformer for Lung Sound Classification
      (DOI 10.21437/Interspeech.2024-106)

Skeleton: the paper states 'experimental setup mostly follows [13]' (Bae Patch-Mix) + 'AST pre-trained from Bae et al. on ImageNet and Audioset'
      → the toolkit ast_patchmix.PatchMixAST (IN+AS ckpt audioset_0.4593.pth, including the swapped 798×128 quirk) is used as the parent as-is.
      Backbone (self.v.*) and mlp_head are fully frozen; only the adapters + classifier (LN+Linear, Eq. 4) are trained.

In each timm Block (12 total) (paper Fig. 1, Eq. 5-8):
  (a) Attention-Adapter — in series after the MHSA output ('positioned after the MHSA layer'):
        h  = W_down(s1·X)                       [768→16]
        h' = h + (1/3)·Σ_{k=3,5,7} DWConv_k(h)
        y  = W_up[ s2·GELU(Conv_1x1(h') + h') ] + s1·X
      The prototype is reference [22] Mona (MonaOp: mean of 2D DWConv 3/5/7 + identity → 1×1 conv + identity → GELU → up-proj).
  (b) Res-Adapter — parallel to the MLP (AdaptFormer-style): W_up(GELU(W_down(X))), bottleneck ratio 0.15 → round(768·0.15)=115.
      In the center panel of Fig. 1 (PDF vector geometry), the Res-Adapter input line branches from the residual stream **after** the MHSA
      residual sum ⊕ and **before** the second LayerNorm, and its output enters the ⊕ after the MLP → input = x (not norm2(x)). The figure
      does not draw the MLP identity skip, but following Eq. (3) and keeping the frozen AST intact, it is implemented as x + MLP(norm2 x) + ResAdapter(x).

Everything the paper leaves unspecified is a flag; the defaults are **one of** the combinations that simultaneously match the parameter counts
(Table 1/2: Attn only 0.34M · Res only 2.14M · both 2.48M, ±0.01M) (per-combination table pinned in tests/unit/models/test_ast_lungadapter.py).
Other combinations such as Res 115 without bias + LN (2,142,724) / Attn 1D + LN + vector s1 (343,300) also satisfy all three values within ±0.01M,
so parameter counts alone cannot determine 1D/2D, scale form, or presence of LN/bias (unresolved without the authors):
  conv_dim      : "2d" (default; patch grid freq 12 × time 79, cls/dist tokens bypass DWConv) | "1d" (token axis, cls/dist included)
  scale_form    : "vector" (default; learned s1∈R^768, s2∈R^16) | "scalar" (learned scalar s1, s2)
  attn_insert   : "pre_residual" (default; x = x + Adapter(attn(norm1 x))) | "post_residual" (Mona code: x = x + attn(...); x = Adapter(x))
  res_input     : "x" (default; Fig. 1 branch point = residual stream after ⊕, before LN2) | "norm2" (same norm2(x) input as the MLP — alternative flag)
  res_scale     : "none" (default; exactly paper Eq. (8) and Fig. 1 right (FC-GELU-FC, no S symbol unlike the left), constant res_scale_init=1)
                  | "vector" (learned 768-d scale) | "scalar" (learned scalar)
                  Note: "vector" is a parameter-count-fitting guess that brings Res-only 2,134,888 (2.13M) and both 2,468,392 (2.47M) to the
                    2.14M/2.48M of Table 1/2, so it is not made the default. The default combination gives Attn only 0.34 ✓ / Res only 2.13
                    (0.0051 off 2.14, within ±0.01M) / both 2.468 (0.0116 off 2.48, exceeding ±0.01M by 0.0016) — why the paper's count is
                    ≈770 higher per block (≈9k total) is unresolved. To match all three values use model.params.res_scale=vector.
  zero_init_up  : True (default; W_up of both adapters initialized to 0 → output identical to the frozen AST at the start of training, AdaptFormer/LoRA convention)
  s1/s2 init 1.0, res_scale init 1.0 (Mona gammax=1 / not specified in the paper — assumed)
  no adapter dropout (the Mona prototype's dropout 0.1 is not adopted; not specified in the paper — assumed)
  weight decay (optimizer wd 1e-6, assumed inherited from Bae) also applies to the s1/s2 scale parameters — no basis in the paper, negligible effect (no flag)
Trainable parameters of the default combination (2D, vector s1/s2, res_scale none): Attn only 338,116 / Res only 2,134,888 / both 2,468,392
(Total = PatchMixAST 87,531,736 + adapters = 87,865,240 / 89,662,012 / 89,995,516) — compared with the paper's 0.34M/2.14M/2.48M
(Total 87.87/89.67/90.01M is the rounded sum 87.53 + Trainable), only 'both' is 0.0116M short (see res_scale above).
With res_scale=vector: 338,116 / 2,144,104 / 2,477,608 (Total 90,004,732), all three within ±0.01M.

Preprocessing wording: paper 3.1.2 says '25 ms Hamming window' vs the inherited icbhi_patchmix (kaldi fbank window_type='hanning', identical to
Bae util/icbhi_util.py:395) — hanning is used, inherited from the Bae code (inconsistent with the paper's wording).

Training recipe (stated in the paper): Adam lr 5e-4, cosine, bs 8, 50 epochs, weighted CE. The rest (SpecAugment icbhi_ast_sup, MA 0.5, AMP, wd 1e-6,
best-on-test highest Score && Se>5) is **assumed** inherited from Bae → adjust in configs/lungadapter_icbhi.yaml and callbacks/lungadapter_paper.yaml.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ast_patchmix import PatchMixAST
from .factory import ModelFactory

logger = logging.getLogger(__name__)


def _make_scale(form: str, dim: int, init: float) -> nn.Parameter:
    if form == "vector":
        return nn.Parameter(torch.full((dim,), float(init)))
    if form == "scalar":
        return nn.Parameter(torch.tensor(float(init)))
    raise ValueError(f"scale form must be vector|scalar, got {form!r}")


class MultiScaleDWConv(nn.Module):
    """Conv part of Eq. (6)-(7) = Mona MonaOp: (1/3)Σ DWConv_{3,5,7} + identity → Conv1×1 + identity.

    conv_dim="2d": input is a (B, C, H, W) grid / "1d": input is a (B, C, N) token sequence. Returns the pre-GELU value (Conv(h')+h')."""

    def __init__(self, channels: int, conv_dim: str = "2d", kernels: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        if conv_dim not in {"1d", "2d"}:
            raise ValueError(f"conv_dim must be 1d|2d, got {conv_dim!r}")
        self.conv_dim = conv_dim
        Conv = nn.Conv2d if conv_dim == "2d" else nn.Conv1d
        self.dwconvs = nn.ModuleList(
            [Conv(channels, channels, kernel_size=k, padding=k // 2, groups=channels) for k in kernels]
        )
        self.projector = Conv(channels, channels, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_prime = h + sum(conv(h) for conv in self.dwconvs) / float(len(self.dwconvs))   # Eq. (6)
        return self.projector(h_prime) + h_prime                                          # inside the brackets of Eq. (7)


class AttentionAdapter(nn.Module):
    """Paper Eq. (5)-(7). forward(x, grid): x (B, N, D); grid=(H, W) is the patch grid in 2D mode (N = n_prefix + H·W)."""

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 16,
        conv_dim: str = "2d",
        scale_form: str = "vector",
        s1_init: float = 1.0,
        s2_init: float = 1.0,
        n_prefix_tokens: int = 2,
        zero_init_up: bool = True,
    ):
        super().__init__()
        self.dim, self.hidden, self.conv_dim, self.n_prefix = dim, hidden, conv_dim, int(n_prefix_tokens)
        self.s1 = _make_scale(scale_form, dim, s1_init)
        self.s2 = _make_scale(scale_form, hidden, s2_init)
        self.down = nn.Linear(dim, hidden)
        self.conv = MultiScaleDWConv(hidden, conv_dim)
        self.up = nn.Linear(hidden, dim)
        if zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def _conv_tokens(self, h: torch.Tensor, grid: Optional[Tuple[int, int]]) -> torch.Tensor:
        b, n, c = h.shape
        if self.conv_dim == "1d":
            # 1D DWConv along the token axis — cls/dist tokens included in the sequence (first 2 positions)
            return self.conv(h.transpose(1, 2)).transpose(1, 2)
        if grid is None:
            raise ValueError("conv_dim='2d' requires the patch grid grid=(H, W)")
        hh, ww = grid
        p = self.n_prefix
        if n != p + hh * ww:
            raise ValueError(f"token count {n} ≠ prefix {p} + {hh}×{ww}")
        patches = h[:, p:, :].reshape(b, hh, ww, c).permute(0, 3, 1, 2)
        patches = self.conv(patches).permute(0, 2, 3, 1).reshape(b, hh * ww, c)
        if p == 0:
            return patches
        # cls/dist tokens: not on the grid, so the DWConv term is 0 (h'=h); only the 1×1 conv (per-token linear) is applied
        prefix = h[:, :p, :]
        prefix = self.conv.projector(prefix.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2) + prefix
        return torch.cat([prefix, patches], dim=1)

    def forward(self, x: torch.Tensor, grid: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        sx = self.s1 * x                                   # s1·X
        h = self.down(sx)                                  # Eq. (5)
        z = self._conv_tokens(h, grid)                     # Conv(h') + h'
        return self.up(self.s2 * F.gelu(z)) + sx           # Eq. (7)


class ResAdapter(nn.Module):
    """Paper Eq. (8): W_up(GELU(W_down(X))). bottleneck = round(dim · ratio) (768·0.15 = 115.2 → 115)."""

    def __init__(self, dim: int = 768, ratio: float = 0.15, bottleneck: Optional[int] = None, zero_init_up: bool = True):
        super().__init__()
        self.bottleneck = int(bottleneck) if bottleneck is not None else int(round(dim * ratio))
        self.down = nn.Linear(dim, self.bottleneck)
        self.up = nn.Linear(self.bottleneck, dim)
        if zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


class LungAdapterBlock(nn.Module):
    """timm Block wrapper. Calls only norm1/attn/ls1/drop_path1/norm2/mlp/ls2/drop_path2 of the wrapped Block (minimizes version dependence).

    timm 1.0.28 Block.forward: x = x + dp1(ls1(attn(norm1 x))); x = x + dp2(ls2(mlp(norm2 x)))  (DeiT: ls/dp = Identity)"""

    def __init__(
        self,
        block: nn.Module,
        attn_adapter: Optional[AttentionAdapter],
        res_adapter: Optional[ResAdapter],
        attn_insert: str = "pre_residual",
        res_input: str = "x",
        res_scale: str = "none",
        res_scale_init: float = 1.0,
        dim: int = 768,
    ):
        super().__init__()
        if attn_insert not in {"pre_residual", "post_residual"}:
            raise ValueError(f"attn_insert must be pre_residual|post_residual, got {attn_insert!r}")
        if res_input not in {"x", "norm2"}:
            raise ValueError(f"res_input must be norm2|x, got {res_input!r}")
        if res_scale not in {"vector", "scalar", "none"}:
            raise ValueError(f"res_scale must be vector|scalar|none, got {res_scale!r}")
        self.block = block
        self.attn_adapter = attn_adapter
        self.res_adapter = res_adapter
        self.attn_insert, self.res_input, self.res_scale_form = attn_insert, res_input, res_scale
        if res_adapter is not None and res_scale != "none":
            self.res_scale = _make_scale(res_scale, dim, res_scale_init)
        else:
            self.register_buffer("res_scale", torch.tensor(float(res_scale_init)), persistent=False)
        self.grid: Optional[Tuple[int, int]] = None   # set by forward_features on every call (H=freq patches, W=time patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blk = self.block
        ls1, dp1 = getattr(blk, "ls1", nn.Identity()), getattr(blk, "drop_path1", nn.Identity())
        ls2, dp2 = getattr(blk, "ls2", nn.Identity()), getattr(blk, "drop_path2", nn.Identity())

        a = blk.attn(blk.norm1(x))
        if self.attn_adapter is not None and self.attn_insert == "pre_residual":
            a = self.attn_adapter(a, self.grid)
        x = x + dp1(ls1(a))
        if self.attn_adapter is not None and self.attn_insert == "post_residual":
            x = self.attn_adapter(x, self.grid)

        n2 = blk.norm2(x)
        m = dp2(ls2(blk.mlp(n2)))
        if self.res_adapter is not None:
            r = self.res_adapter(n2 if self.res_input == "norm2" else x)
            m = m + self.res_scale * r
        return x + m


@ModelFactory.register_model_class("ast_lungadapter")
class LungAdapterAST(PatchMixAST):
    """PatchMixAST (IN+AS, swapped 798×128 quirk) + per-block LungAdapter, frozen backbone. method is fixed to 'ce'."""

    def __init__(
        self,
        n_cls: int = 4,
        # ---- adapters (stated in the paper) ----
        use_attn_adapter: bool = True,
        use_res_adapter: bool = True,
        attn_hidden: int = 16,
        res_ratio: float = 0.15,
        res_bottleneck: Optional[int] = None,
        # ---- not specified in the paper → flags (see module docstring) ----
        conv_dim: str = "2d",
        scale_form: str = "vector",
        s1_init: float = 1.0,
        s2_init: float = 1.0,
        attn_insert: str = "pre_residual",
        res_input: str = "x",
        res_scale: str = "none",
        res_scale_init: float = 1.0,
        zero_init_up: bool = True,
        freeze_backbone: bool = True,
        # ---- parent (PatchMixAST) arguments ----
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 798,
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        input_layout: str = "freq_time",
    ):
        super().__init__(
            n_cls=n_cls, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain, model_size=model_size,
            pretrained_path=pretrained_path, verbose=verbose, method="ce", input_layout=input_layout,
        )
        if not (use_attn_adapter or use_res_adapter):
            logger.warning("LungAdapterAST: no adapters — training only the frozen AST + head (linear probe)")
        dim = self.original_embedding_dim
        self.adapter_cfg: Dict[str, object] = dict(
            use_attn_adapter=use_attn_adapter, use_res_adapter=use_res_adapter, attn_hidden=attn_hidden,
            res_ratio=res_ratio, res_bottleneck=res_bottleneck, conv_dim=conv_dim, scale_form=scale_form,
            attn_insert=attn_insert, res_input=res_input, res_scale=res_scale, zero_init_up=zero_init_up,
        )

        wrapped = []
        for blk in self.v.blocks:
            attn_ad = AttentionAdapter(
                dim, attn_hidden, conv_dim=conv_dim, scale_form=scale_form, s1_init=s1_init, s2_init=s2_init,
                n_prefix_tokens=2, zero_init_up=zero_init_up,
            ) if use_attn_adapter else None
            res_ad = ResAdapter(dim, res_ratio, res_bottleneck, zero_init_up=zero_init_up) if use_res_adapter else None
            wrapped.append(LungAdapterBlock(blk, attn_ad, res_ad, attn_insert, res_input, res_scale, res_scale_init, dim))
        self.v.blocks = nn.Sequential(*wrapped)

        # actual patch grid: forward_features applies the conv after transpose(2,3), so (H=freq patches, W=time patches) = (12, 79)
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)   # (79, 12) — in terms of the quirk arguments
        self.token_grid = (t_dim, f_dim)
        self._set_grid(self.token_grid)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze()
        tr, tot = self.count_parameters()
        logger.info("LungAdapterAST: trainable %.3fM / total %.3fM (%s)", tr / 1e6, tot / 1e6, self.adapter_cfg)

    # ------------------------------------------------------------------ utilities
    def _set_grid(self, grid: Tuple[int, int]) -> None:
        for blk in self.v.blocks:
            blk.grid = grid

    def _freeze(self) -> None:
        """Freeze self.v.* (including the wrapped Blocks) and mlp_head; train only adapters, scales, and classifier."""
        for p in self.v.parameters():
            p.requires_grad = False
        for p in self.mlp_head.parameters():
            p.requires_grad = False
        for blk in self.v.blocks:
            for name, p in blk.named_parameters():
                if not name.startswith("block."):
                    p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def adapter_parameters(self):
        for blk in self.v.blocks:
            for name, p in blk.named_parameters():
                if not name.startswith("block."):
                    yield p

    def count_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ---------------------------------------------------------------- forward
    def forward_features(self, x, y=None, patch_mix=False, time_domain=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # same grid computation as the parent (x before transpose: (B,1,T,F) → grid H=(F-16)/10+1, W=(T-16)/10+1)

        grid = (int((x.size(3) - 16) / 10) + 1, int((x.size(2) - 16) / 10) + 1)
        self._set_grid(grid)
        return super().forward_features(x, y=y, patch_mix=patch_mix, time_domain=time_domain)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        return super().training_forward(audio_batch, labels, loss_func)
