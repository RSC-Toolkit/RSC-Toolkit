# Portions of this file are adapted from VMamba (https://github.com/MzeroMiko/VMamba); that code is under the MIT license.
# Portions of this file are adapted from DASS (https://github.com/Saurabhbhati/DASS); that code is under the BSD-3-Clause license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
VMamba VSSM (v2 family, DASS-medium setting) — faithful port of the Lung-SRAD official code models_DASS/vmamba_patchmix.py
(exact port of the official code, verified against goldens).

Source: RSC-Toolkit/Lung-SRAD d28d783 models_DASS/vmamba_patchmix.py (upstream MzeroMiko/VMamba MIT, Saurabhbhati/DASS BSD-3).
Port scope = only the paths DASS-medium actually uses:
  patchembed v2 · downsample v3 · LN2D(channel_first) · SS2D forward_type "v05_noz"(forwardv2/forward_corev2, no_einsum,
  force_fp32=False, oflex) · ssm_d_state 1 · ssm_ratio 1.0 · ssm_conv 3(bias False) · mlp_ratio 4 GELU · DropPath 0.3 ·
  classifier norm→avgpool→flatten→head.
Non-structural differences from the original:
  - einops.repeat → torch (A_log_init/D_init), timm DropPath/trunc_normal_ → torch implementations (same definitions).
  - CrossScan/CrossMerge use pure torch instead of triton (csms6s version, same layout). Selective scan uses the kernel if available, otherwise the torch fallback.
  - SS2Dv0/SS2Dv3 (xv*), cascade2d, out_norm variants (_onnone/_ondwconv3/...), posembed, use_checkpoint, flops are unused and omitted.
  - The three patch-mix modes (2d/time/freq) choose between the code behavior (as_code) and the paper definition (as_paper) via `mix_layout` (see below).

Gaussian blur hook (Spectral-Aware Regularization): in SS2D.forward_corev2, after in_proj→dwconv→SiLU and right before CrossScan, on x(B,D,H,W):
  x = strength · blur_separable(x; K, σ)  (ℓ1-normalized 1-D Gaussian as vertical and horizontal depthwise convs, no parameters, always applied in training and evaluation)
  Applied blocks = stage ∈ blur_stage and block index (0-based) ∈ blur_blocks (original _make_layer: blur_on = stage_id in blur_stage and d in blur_blocks).

Patch-Mix (on the tensor right after patch_embed, replacing a contiguous span with the batch randperm index; λ from np.random.beta(β,β)):
  The tensor is channel_first, i.e. actually (B, C=96, F=32, T=256), but the original function assumes (B,H,W,C) via `B, H, W, C = x.shape`.
  - as_code (default, matches the official code): original behavior as-is —
      'time': replaces a contiguous span of dim2 (= frequency, 32 bins), num_mask = max(1, int(32·(1−λ))), λ ← 1 − num_mask/32
      'freq': replaces a contiguous span of dim1 (= channels, 96), num_mask = max(1, int(96·(1−λ))), λ ← 1 − num_mask/96
      '2d'  : of the (dim1×dim2)=3072 'tokens' (each token = a time vector of 256), int(3072·(1−λ)) are replaced via randperm, λ ← 1 − num_mask/3072
      The time axis (256) is never partially replaced in any mode.
  - as_paper (Eq.6-7): 'time' uses width max(1, int(W(1−λ))) on the time axis dim3 (W=256), 'freq' uses height max(1, int(H(1−λ))) on the
      frequency axis dim2 (H=32), '2d' replaces (F×T)=8192 patches (each patch = a channel vector of 96). The RNG call order
      (beta → randperm(B) → randint/randperm) is the same as as_code.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_scan import CrossMerge, CrossScan
from .selective_scan import selective_scan


# =====================================================================================
# small components (vmamba_patchmix.py:45-131)
# =====================================================================================
class Linear2d(nn.Linear):
    """Linear for channel_first tensors — applied as a 1×1 conv (weights keep the nn.Linear shape)."""

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + "weight" in state_dict:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Same definition as timm.layers.DropPath(scale_by_keep=True): per-sample Bernoulli(keep) mask / keep."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# =====================================================================================
# mamba initialization (vmamba_patchmix.py:168-224)
# =====================================================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, d_state).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


def gaussian_kernel1d(kernel_size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """vmamba_patchmix.py:552-556 — ℓ1-normalized 1-D Gaussian (K=5, σ=3 → [0.1782, 0.2105, 0.2225, 0.2105, 0.1782])."""
    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = g / (g.sum() + 1e-12)
    return g.to(dtype)


def blur_separable(x: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """vmamba_patchmix.py:558-566 — depthwise separable Gaussian blur (vertical → horizontal), no parameters."""
    B, D, H, W = x.shape
    g = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    w_h = g.view(1, 1, kernel_size, 1).expand(D, 1, kernel_size, 1).contiguous()
    w_w = g.view(1, 1, 1, kernel_size).expand(D, 1, 1, kernel_size).contiguous()
    pad = kernel_size // 2
    x = F.conv2d(x, w_h, padding=(pad, 0), groups=D)
    x = F.conv2d(x, w_w, padding=(0, pad), groups=D)
    return x


# =====================================================================================
# SS2D (v2 family only: vmamba_patchmix.py:380-750)
# =====================================================================================
class SS2D(nn.Module, mamba_init):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        # blur hook
        blur_enable=False,
        blur_kernel_size=5,
        blur_sigma=1.0,
        blur_strength=1.0,
        # selective scan path
        ssm_backend: str = "auto",
        **kwargs,
    ):
        super().__init__()
        if forward_type.startswith("xv") or forward_type in ("v0", "v0seq"):
            raise NotImplementedError(f"forward_type {forward_type!r} is outside the port scope (v2 family only)")
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.blur_enable = bool(blur_enable)
        self.blur_kernel_size = int(blur_kernel_size)
        self.blur_sigma = float(blur_sigma)
        self.blur_strength = float(blur_strength)
        self.ssm_backend = ssm_backend
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        for tag in ("_onnone", "_ondwconv3", "_oncnorm", "_onsoftmax", "_onsigmoid"):
            hit, forward_type = checkpostfix(tag, forward_type)
            if hit:
                raise NotImplementedError(f"out_norm variant {tag} is outside the port scope")
        self.out_norm = LayerNorm(d_inner)

        # forward_type → forward_corev2 arguments (original FORWARD_TYPES; CrossScan is always the torch version, SelectiveScan picks a backend)
        FORWARD_TYPES = dict(
            v01=dict(force_fp32=(not self.disable_force32), no_einsum=False),
            v02=dict(force_fp32=(not self.disable_force32), no_einsum=False),
            v03=dict(force_fp32=(not self.disable_force32), no_einsum=False),
            v04=dict(force_fp32=False, no_einsum=False),
            v05=dict(force_fp32=False, no_einsum=True),
            v2=dict(force_fp32=(not self.disable_force32), no_einsum=False),
            v3=dict(force_fp32=False, no_einsum=False),
        )
        if forward_type not in FORWARD_TYPES:
            raise NotImplementedError(f"forward_type {forward_type!r} is outside the port scope")
        self.forward_type_tag = forward_type
        self._core_kwargs = FORWARD_TYPES[forward_type]
        self.forward_core = partial(self.forward_corev2, **self._core_kwargs)
        k_group = 4

        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2,
            )

        self.x_proj = [nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            self.dt_projs = [self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(k_group)]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        else:
            raise NotImplementedError(initialize)

    def blur_separable(self, x, kernel_size=7, sigma=1.5):
        return blur_separable(x, kernel_size, sigma)

    def forward_corev2(self, x: torch.Tensor, to_dtype=True, force_fp32=False, ssoflex=True, no_einsum=False, **kwargs):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if self.blur_enable:  # vmamba_patchmix.py:682-684 — right before CrossScan (training and evaluation alike)
            x = self.blur_strength * self.blur_separable(x, kernel_size=self.blur_kernel_size, sigma=self.blur_sigma)

        xs = CrossScan.apply(x)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                             bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).contiguous()
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = (t.to(torch.float32) for t in (xs, dts, Bs, Cs))

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, oflex=ssoflex, backend=self.ssm_backend,
        ).view(B, K, -1, H, W)
        y: torch.Tensor = CrossMerge.apply(ys)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y)
        return y.to(x.dtype) if to_dtype else y

    def forward(self, x: torch.Tensor, **kwargs):
        """forwardv2 (vmamba_patchmix.py:732-749)."""
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# =====================================================================================
# VSSBlock / VSSM (vmamba_patchmix.py:1066-1586)
# =====================================================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer=nn.LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        blur_enable=False,
        blur_kernel_size=5,
        blur_sigma=1.0,
        blur_strength=1.0,
        ssm_backend: str = "auto",
        **kwargs,
    ):
        super().__init__()
        if gmlp or use_checkpoint:
            raise NotImplementedError("gmlp / use_checkpoint are outside the port scope")
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank, act_layer=ssm_act_layer,
                d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate, initialize=ssm_init,
                blur_enable=blur_enable, blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma, blur_strength=blur_strength,
                forward_type=forward_type, channel_first=channel_first, ssm_backend=ssm_backend,
            )
        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio), act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


_NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
_ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

MIX_LAYOUTS = ("as_code", "as_paper")
MIX_TYPES = ("2d", "time", "freq")


class VSSM(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=(2, 2, 9, 2),
        dims=(96, 192, 384, 768),
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",
        downsample_version: str = "v2",
        patchembed_version: str = "v1",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
        # patch-mix
        enable_patch_mix=False,
        mix_beta=1.0,
        mix_layout: str = "as_code",
        # blur
        blur_blocks: Iterable[int] = (2, 3, 12),
        blur_stage: Iterable[int] = (2,),
        kernel_size=5,
        sigma=1.0,
        blur_strength=1.0,
        ssm_backend: str = "auto",
        **kwargs,
    ):
        super().__init__()
        if posembed:
            raise NotImplementedError("posembed is outside the port scope")
        if mix_layout not in MIX_LAYOUTS:
            raise ValueError(f"mix_layout must be one of {MIX_LAYOUTS}, got {mix_layout!r}")
        self.enable_patch_mix = enable_patch_mix
        self.mix_beta = float(mix_beta)
        self.mix_layout = mix_layout
        self.channel_first = norm_layer.lower() in ["bn", "ln2d"]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        depths = list(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        dims = list(dims)
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        blur_blocks = set(int(b) for b in blur_blocks)
        blur_stage = set(int(s) for s in blur_stage)
        self.blur_config = dict(blur_blocks=sorted(blur_blocks), blur_stage=sorted(blur_stage), kernel_size=int(kernel_size),
                                sigma=float(sigma), blur_strength=float(blur_strength))

        norm_layer_cls = _NORMLAYERS[norm_layer.lower()]
        ssm_act_layer_cls = _ACTLAYERS[ssm_act_layer.lower()]
        mlp_act_layer_cls = _ACTLAYERS[mlp_act_layer.lower()]
        self.pos_embed = None

        _make_patch_embed = dict(v1=self._make_patch_embed, v2=self._make_patch_embed_v2)[patchembed_version]
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer_cls, channel_first=self.channel_first)

        _make_downsample = dict(v2=self._make_downsample, v3=self._make_downsample_v3, none=(lambda *_, **_k: None))[downsample_version]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], self.dims[i_layer + 1], norm_layer=norm_layer_cls, channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            self.layers.append(self._make_layer(
                dim=self.dims[i_layer], stage_id=i_layer,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint, norm_layer=norm_layer_cls, downsample=downsample, channel_first=self.channel_first,
                blur_blocks=blur_blocks, blur_stage=blur_stage,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank, ssm_act_layer=ssm_act_layer_cls,
                ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias, ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init,
                forward_type=forward_type, mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer_cls, mlp_drop_rate=mlp_drop_rate,
                kernel_size=kernel_size, sigma=sigma, blur_strength=blur_strength, gmlp=gmlp, ssm_backend=ssm_backend,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer_cls(self.num_features),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, drop_path=(0.1, 0.1), use_checkpoint=False, norm_layer=nn.LayerNorm, downsample=nn.Identity(), channel_first=False,
        stage_id=0, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
        ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0, gmlp=False,
        blur_blocks=None, blur_stage=None, kernel_size=5, sigma=1.0, blur_strength=1.0, ssm_backend="auto", **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blur_on = (stage_id in blur_stage) and (d in blur_blocks)   # vmamba_patchmix.py:1404
            blur_kwargs = dict(blur_enable=True, blur_kernel_size=kernel_size, blur_sigma=sigma, blur_strength=blur_strength) if blur_on else {}
            blocks.append(VSSBlock(
                hidden_dim=dim, drop_path=drop_path[d], norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank, ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias, ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init,
                forward_type=forward_type, mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer, mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp, use_checkpoint=use_checkpoint, ssm_backend=ssm_backend, **blur_kwargs,
            ))
        return nn.Sequential(OrderedDict(blocks=nn.Sequential(*blocks), downsample=downsample))

    # ------------------------------------------------------------------ patch-mix
    def _draw_lam(self) -> float:
        return float(np.random.beta(self.mix_beta, self.mix_beta)) if self.mix_beta > 0 else 1.0

    def _mix_axis(self, x: torch.Tensor, target, axis: int):
        """Replace a contiguous span of axis `axis` with x[index] (shared skeleton of the original time_only/freq_only; RNG order beta→randperm(B)→randint)."""
        lam = self._draw_lam()
        B = x.shape[0]
        size = x.shape[axis]
        device = x.device
        index = torch.randperm(B).to(device)
        num_mask = max(1, int(size * (1.0 - lam)))
        start = torch.randint(0, size - num_mask + 1, (1,), device=device).item()
        sl = [slice(None)] * x.dim()
        sl[axis] = slice(start, start + num_mask)
        sl = tuple(sl)
        x[sl] = x[index][sl]
        lam = 1.0 - num_mask / size
        return x, target, target[index], lam, index

    def patch_mix_time_only(self, x, target):
        # as_code: original `B,H,W,C = x.shape` → W = dim2 (actually frequency, 32).  as_paper: time axis dim3 (Eq.6)
        return self._mix_axis(x, target, axis=2 if self.mix_layout == "as_code" else 3)

    def patch_mix_freq_only(self, x, target):
        # as_code: original H = dim1 (actually channels, 96).  as_paper: frequency axis dim2 (Eq.7)
        return self._mix_axis(x, target, axis=1 if self.mix_layout == "as_code" else 2)

    def patch_mix_2d_ast_equiv(self, x, target):
        lam = self._draw_lam()
        device = x.device
        B = x.shape[0]
        index = torch.randperm(B).to(device)
        if self.mix_layout == "as_code":
            # original: B,H,W,C = x.shape gives (channels×frequency)=N tokens, each token = a time vector
            _, H, W, C = x.shape
            N = H * W
            x_flat = x.reshape(B, N, C)
            num_mask = int(N * (1.0 - lam))
            mask = torch.randperm(N, device=device)[:num_mask]
            x_flat[:, mask, :] = x_flat[index][:, mask, :]
            lam = 1.0 - num_mask / N
            x = x_flat.view(B, H, W, C)
        else:
            # as_paper: (F×T) patches, each patch = a channel vector
            _, C, Fh, T = x.shape
            N = Fh * T
            x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
            num_mask = int(N * (1.0 - lam))
            mask = torch.randperm(N, device=device)[:num_mask]
            x_flat[:, mask, :] = x_flat[index][:, mask, :]
            lam = 1.0 - num_mask / N
            x = x_flat.view(B, Fh, T, C).permute(0, 3, 1, 2)
        return x, target, target[index], lam, index

    def patch_mix_dual_axis(self, x, target):
        """
        Verbatim port of the **dead code** `patch_mix_dual_axis` in the original `vmamba_patchmix.py:1524`.

        In forward, the `elif mix_type == "dual":` branch is commented out with `'''...'''` and preceded by
        `else: raise ValueError`, so it **never runs in the public code**. The toolkit's default dual path
        follows the actual behavior of `main.py:338-352` (**two independent forwards** for time and frequency **+ two contrastive losses**).

        This function is a **variant for testing the possibility** that the unreleased `main_v2.py` revived this dead code
        (`cl_mode=dual_seq`). It keeps three characteristics of the original:
          - Time-axis mix is applied first, then frequency-axis mix is applied **on top of that result** with new random pairs (sequential).
          - The effective λ is the product of the two λ (`lam_t * lam_f`) — original comment "approximate effective mixing ratio".
          - **Only the time-axis mix labels are used** (`target_a`, `target_b`, `index_t`). The pairing information of the
            frequency-axis mix (`target_b2`, `index_f`) is simply discarded by the original — a quirk where inputs are mixed but labels are not.
        """
        x, target_a, target_b, lam_t, index_t = self.patch_mix_time_only(x, target)
        x, _target_a2, _target_b2, lam_f, _index_f = self.patch_mix_freq_only(x, target)
        return x, target_a, target_b, lam_t * lam_f, index_t

    def apply_patch_mix(self, x, y, mix_type: str):
        if mix_type == "2d":
            return self.patch_mix_2d_ast_equiv(x, y)
        if mix_type == "time":
            return self.patch_mix_time_only(x, y)
        if mix_type == "freq":
            return self.patch_mix_freq_only(x, y)
        if mix_type == "dual_seq":
            return self.patch_mix_dual_axis(x, y)
        raise ValueError(f"Invalid mix_type {mix_type!r} (2d|time|freq|dual_seq)")

    # ------------------------------------------------------------------ forward
    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier.norm(x)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2)
        x = self.classifier.avgpool(x)
        return torch.flatten(x, 1)

    def forward_features(self, x, y=None, patch_mix: bool = False, mix_type: str = "none"):
        """vmamba_patchmix.py:1541-1586 — returns (feat, y_a, y_b, lam, index) with patch_mix, otherwise (features, logits)."""
        x = self.patch_embed(x)
        if patch_mix:
            x, y_a, y_b, lam, index = self.apply_patch_mix(x, y, mix_type)
        for layer in self.layers:
            x = layer(x)
        if patch_mix:
            return self.pool_features(x), y_a, y_b, lam, index
        features = self.pool_features(x)
        logits = self.classifier(x)
        return features, logits

    def forward(self, x, y=None, patch_mix: bool = False, mix_type: str = "none"):
        return self.forward_features(x, y=y, patch_mix=patch_mix, mix_type=mix_type)


# =====================================================================================
# DASS-medium setup + ckpt loading (ast_models.py:313-336, 350-359)
# =====================================================================================
DASS_MEDIUM_KWARGS = dict(
    drop_path_rate=0.3, embed_dim=96, depths=[2, 2, 20, 2], ssm_d_state=1, ssm_ratio=1.0, ssm_conv_bias=False,
    forward_type="v05_noz", downsample_version="v3", patchembed_version="v2", norm_layer="LN2D", num_classes=1000,
)


def build_dass_medium_vssm(blur: bool = True, blur_blocks=(2, 3), blur_stage=(2,), kernel_size=5, sigma=3.0, blur_strength=1.0,
                           enable_patch_mix=False, mix_beta=1.0, mix_layout="as_code", ssm_backend="auto",
                           drop_path_rate: float = 0.3) -> VSSM:
    """VSSM(medium) of ast_models.py:313-317 — blur=False gives blur_blocks=∅ (same structure as FT/plain DASS)."""
    kw = dict(DASS_MEDIUM_KWARGS)
    kw["drop_path_rate"] = drop_path_rate
    embed = kw.pop("embed_dim")
    depths = kw.pop("depths")
    return VSSM(
        depths=depths, dims=[embed * 2 ** i for i in range(len(depths))],
        enable_patch_mix=enable_patch_mix, mix_beta=mix_beta, mix_layout=mix_layout,
        blur_blocks=(blur_blocks if blur else ()), blur_stage=blur_stage, kernel_size=kernel_size, sigma=sigma,
        blur_strength=blur_strength, ssm_backend=ssm_backend, **kw,
    )


def strip_dass_state_dict(state_dict: dict) -> dict:
    """ast_models.py:353-358 — strip the 'module.v.' prefix, exclude classifier.head."""
    out = {}
    for k, v in state_dict.items():
        if "classifier.head" in k:
            continue
        out[k.replace("module.v.", "")] = v
    return out


def load_dass_checkpoint(vssm: VSSM, path: str, expect_missing: Sequence[str] = ("classifier.head.weight", "classifier.head.bias")):
    """Load DASS_medium_v2.pth — the original uses strict=False, but here it checks that there are no missing/unexpected keys other than the 2 head keys."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and all(not k.startswith("module.") for k in raw):
        raw = raw["model"]
    sd = strip_dass_state_dict(raw)
    result = vssm.load_state_dict(sd, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    if missing != sorted(expect_missing) or unexpected:
        raise RuntimeError(f"DASS ckpt key mismatch: missing={missing}, unexpected={unexpected}")

    return {"loaded": len(sd), "missing": missing, "unexpected": unexpected}
