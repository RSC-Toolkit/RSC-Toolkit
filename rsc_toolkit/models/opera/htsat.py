# Portions of this file are adapted from OPERA (https://github.com/evelyn0414/OPERA), HTS-AT and Swin Transformer; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Toolkit re-implementation of HTSAT-Swin (OPERA-CT encoder) — faithful reproduction of evelyn0414/OPERA src/model/htsat/htsat.py (Ke Chen's HTS-AT with the torchlibrosa front end commented out).

Relation to the original code:
- Module names, buffers (attn_mask, relative_position_index) and parameter layout match the original HTSAT_Swin_Transformer, so the
  `encoder.encoder.htsat.*` keys of the official ckpt load strictly after stripping the prefix (197 keys). The unused torchlibrosa front end
  (spectrogram_extractor·logmel_extractor) and SpecAugmentation are not included (the original forward comments them out too).
- Input is an external mel (B,1,T,64). forward: bn0 → (T≤1024) reshape_wav2img (T→1024 bicubic interpolation, 4 pieces stacked along the frequency axis to 256×256)
  → patch_embed(4×4/4) → Swin 4 stages (depth 2,2,6,2 · window 8) → norm → token-semantic reshape, then mean over all tokens = latent_output(768).
  If T>1024: sliding 1024 windows with 512 overlap and latent averaging at eval, random crop at train (same as the original).
- tscam_conv·head (AudioSet 527) are in the ckpt and kept as modules, but only latent_output is used, so they are not computed
  (framewise/clipwise outputs omitted). The eval-mode collection and averaging of attention maps is also omitted, as it does not affect the output.
- DropPath (drop_path_rate 0.1, linspace decay) is active in train mode — the EMBC frozen protocol (freeze_body) also runs under model.train(),
  so it stays active, and bn0 uses batch statistics and updates its running statistics (quirk of the original, kept).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HTSATSwin", "load_opera_ct_checkpoint", "OPERA_CT_DROPPED_KEY_PREFIXES"]


def _to_2tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class PatchEmbed(nn.Module):
    """htsat_util.PatchEmbed — patch embedding whose stride may differ from the kernel (here 4×4/4, padding 0)."""

    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, patch_stride=4):
        super().__init__()
        img_size, patch_size, patch_stride = _to_2tuple(img_size), _to_2tuple(patch_size), _to_2tuple(patch_stride)
        self.img_size, self.patch_size, self.patch_stride = img_size, patch_size, patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"input {H}x{W} ≠ model {self.img_size}"
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        n = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim, self.input_resolution, self.num_heads = dim, input_resolution, num_heads
        self.window_size, self.shift_size, self.mlp_ratio = window_size, shift_size, mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # original: no partitioning when the window is at least the resolution (last stage 8×8 → shift 0)
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=_to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in slices:
                for w in slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution, self.dim = input_resolution, dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim, self.input_resolution, self.depth = dim, input_resolution, depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class HTSATSwin(nn.Module):
    """HTSAT_Swin_Transformer of OPERA-CT (config.py: spec 256, patch 4/4, dim 96, depth 2,2,6,2, heads 4,8,16,32, window 8, mel 64, 527 classes, tscam)."""

    def __init__(self, spec_size=256, patch_size=4, patch_stride=(4, 4), in_chans=1, num_classes=527, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 32), window_size=8, mlp_ratio=4.0, qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, mel_bins=64, patch_norm=True):
        super().__init__()
        self.spec_size, self.patch_stride, self.mel_bins = spec_size, tuple(patch_stride), mel_bins
        self.depths, self.num_layers = list(depths), len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.freq_ratio = spec_size // mel_bins  # 4
        norm_layer = nn.LayerNorm
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.patch_embed = PatchEmbed(img_size=spec_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm else None, patch_stride=patch_stride)
        patches_resolution = self.patch_embed.grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None))
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # for ckpt compatibility (AudioSet token-semantic head) — not used in the latent_output path
        SF = spec_size // (2 ** (len(depths) - 1)) // self.patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(self.num_features, num_classes, kernel_size=(SF, 3), padding=(0, 1))
        self.head = nn.Linear(num_classes, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---- latent_output part of the original forward_features ----
    def forward_features(self, x):
        frames_num = x.shape[2]
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, Fq, T = x.shape
        c_freq_bin = Fq // self.freq_ratio
        x = x.reshape(B, C, Fq // c_freq_bin, c_freq_bin, T).permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        return torch.flatten(self.avgpool(torch.flatten(x, 2)), 1)  # (B, 768)

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            crop_pos = random.randint(0, time_steps - crop_size - 1) if spe_pos is None else spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size, :]
        return tx

    def reshape_wav2img(self, x):
        B, C, T, Fq = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and Fq <= target_F, "the wav size should less than or equal to the swin input size"
        if T < target_T:
            x = F.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if Fq < target_F:
            x = F.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()  # B C F T
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])  # B C 256 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T, mel_bins) external mel → latent_output (B, 768). Returns only the latent_output of the original forward."""
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        if x.shape[2] > self.freq_ratio * self.spec_size:
            if self.training:
                return self.forward_features(self.reshape_wav2img(self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)))
            overlap_size, crop_size = 512, 1024  # hard-coded in the original
            outs = [self.forward_features(self.reshape_wav2img(self.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)))
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size)]
            return torch.stack(outs, 0).float().mean(0)
        return self.forward_features(self.reshape_wav2img(x))


# Keys of the official Cola(htsat) ckpt that the toolkit drops: the unused torchlibrosa front end (inside htsat) and the COLA projector (g, layer_norm, linear —
# extract_feature(dim=768) returns the encoder output as is, so neither protocol uses it)
OPERA_CT_DROPPED_KEY_PREFIXES = (
    "encoder.encoder.htsat.spectrogram_extractor.",
    "encoder.encoder.htsat.logmel_extractor.",
    "g.", "layer_norm.", "linear.",
)
_HTSAT_PREFIX = "encoder.encoder.htsat."


def load_opera_ct_checkpoint(model: HTSATSwin, path: str) -> List[str]:
    """encoder-operaCT.ckpt (Lightning, Cola encoder='htsat') → strict load into HTSATSwin. Returns the list of dropped keys."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    new_sd, dropped = {}, []
    for k, v in sd.items():
        if k.startswith(OPERA_CT_DROPPED_KEY_PREFIXES):
            dropped.append(k)
        elif k.startswith(_HTSAT_PREFIX):
            new_sd[k[len(_HTSAT_PREFIX):]] = v
        else:
            raise KeyError(f"unexpected ckpt key: {k}")
    model.load_state_dict(new_sd, strict=True)
    return dropped
