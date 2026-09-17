"""
Audio-MAE backbone + DDE-MAE (Disentangling Dual-Encoder MAE) — Wei, Miao, Li, Interspeech 2025 (arXiv 2506.10698).

[Official-code parity, verified against goldens — official code exists] Audio-MAE (facebookresearch/AudioMAE, models_mae.py / models_vit.py,
timm 0.3.2 lineage) re-implemented on timm 1.0 Blocks. Matches the goldens (scripts/reproduction/dump_ddemae_goldens.py, separate venv, CPU) within 1e-4:
  - AudioMAEEncoder : ViT-B/16, img_size (1024, 128)=(T, F), patch 16 → grid 64x8=512 tokens (t-major), cls_token,
                      fixed 2D sin-cos pos_embed (official get_2d_sincos_pos_embed_flexible(patch_hw=(8,64)) — the official quirk of computing on
                      an (8,64) grid and attaching it to t-major tokens is kept; verified to match the ckpt values), timm Block x12 (768, heads 12,
                      mlp 4, qkv_bias, LN eps 1e-6), norm (MAE latent) / fc_norm (global_pool fine-tuning, models_vit.py:52-54).
  - MAEDecoder      : decoder_embed Linear(in_dim → 512), mask_token, decoder_pos_embed(513, 512) fixed sin-cos,
                      decoder_mode=1 (used to train the public AS-2M ckpt): AudioMAE timm_patch/swin_transformer.py SwinTransformerBlock x16
                      (dim 512, heads 16, feat (64,8), window (4,4), shift (2,0) on odd blocks, scaled-cosine attn + log-CPB meta MLP, post-norm),
                      operating on the (64,8) grid after removing cls; decoder_mode=0: timm Block x8 (global attention, cls kept then removed).
                      decoder_norm, decoder_pred(512→256).
  - random_masking / patchify / unpatchify / MAE reconstruction loss (models_mae.py:225-247, :171-197, :395-411).
  - load_audiomae_checkpoint: strict load of only name- and shape-matching keys from pretrained.pth['model'] (encoder 150 keys, decoder 280 keys = 430; drop 0).

[Re-implementation from the paper description — no code, no goldens] DDE part (together with rsc_toolkit/methods/dde_mae.py):
  - DDEMAE (model 'audiomae_dde', stage 1): duplicate the pretrained encoder → E_R (original mel), E_I (original mel + time-shuffled mel in parallel).
      Fusion = per-token concat along the feature dimension (read from Fig. 1; 1536 → newly initialized decoder_embed [assumption]) → decoder →
      masked mel reconstruction L_recon (Eq. 5 MSE).
      L_siamese (Eq. 6, margin assumed 1.0, pooled vectors assumed), L_MI = vCLUB (Eq. 4, pooled vectors assumed, auxiliary Adam for q_θ with
      lr 1e-4 assumed, once per step).
      Total loss Eq. 7: L = L_recon + 0.5 L_siamese + 0.02 L_MI. variant='s2' is a single encoder + original decoder reconstruction (L_recon) only.
  - AudioMAEClassifier (model 'audiomae_dde_cls', B2/stage 2): encoder (+ only the stage-1 E_R loaded via encoder_ckpt) → global_pool+fc_norm [assumption] → Linear → CE.

Assumptions (all config flags): mask_ratio 0.8, norm_pix_loss False, decoder_mode 1 (keys match the ckpt),
decoder_embed 1536→512 newly initialized, time-shuffle unit 'patch', masks of the two E_I inputs = physically the same patches, pooled vectors for
Siamese/MI, margin 1.0, q_θ MLP (768-512-768) + auxiliary Adam 1e-4, classification head global_pool (fc_norm), E_I discarded.

quirk: pl_adapter uses a single optimizer, so q_θ is updated by an auxiliary optimizer inside the model (within training_forward). The MI term of
the main loss is computed with a functional call on detached q_θ parameters so the main optimizer never touches q_θ (grad None → AdamW skips it).
The auxiliary optimizer state is not stored in last.ckpt.
"""
from __future__ import annotations

import copy
import logging
import pathlib
from functools import partial
from typing import Dict, Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .base import BaseRSCModel
from .factory import ModelFactory
from .m2d.encoder import PatchEmbed, get_2d_sincos_pos_embed
from rsc_toolkit.paths import resolve_pretrained
from rsc_toolkit.methods.dde_mae import VCLUB, permute_token_noise, siamese_hinge, time_shuffle

logger = logging.getLogger(__name__)

AUDIOMAE_CKPT = "pretrained_models/upstream/audiomae/pretrained.pth"


# =================================================================================================== encoder
class AudioMAEEncoder(nn.Module):
    """Encoder part of the official MaskedAutoencoderViT (models_mae.py) + global_pool/fc_norm of models_vit.py."""

    def __init__(self, img_size=(1024, 128), patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, global_pool: bool = True):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        ps = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.patch_embed = PatchEmbed(tuple(img_size), ps, in_chans, embed_dim)
        gt, gf = self.patch_embed.grid_size  # (T/16, F/16) = (64, 8)
        self.patch_hw = (gf, gt)  # official PatchEmbed_org.patch_hw = (img_size[1]//p, img_size[0]//p) = (8, 64)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.global_pool = bool(global_pool)
        self.fc_norm = norm_layer(embed_dim)  # models_vit.py: new LN used only when global_pool=True (not in the ckpt)
        self.embed_dim = embed_dim
        self.pos_embed.data.copy_(torch.from_numpy(
            get_2d_sincos_pos_embed(embed_dim, self.patch_hw, cls_token=True)).float().unsqueeze(0))
        nn.init.normal_(self.cls_token, std=0.02)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(_init_weights)

    @property
    def grid_t(self) -> int:
        return self.patch_embed.grid_size[0]

    @property
    def grid_f(self) -> int:
        return self.patch_embed.grid_size[1]

    def forward_tokens(self, x: torch.Tensor, mask_ratio: float = 0.0, noise: Optional[torch.Tensor] = None):
        """
        x: (B, 1, T, F). models_mae.forward_encoder: patch_embed → +pos → random_masking → cls → blocks → norm.
        Returns (latent (B, 1+keep, D), mask (B, L) [1=removed], ids_restore (B, L)).
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, mask_ratio, noise)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """models_vit.VisionTransformer.forward_features: global_pool → mean(patches) → fc_norm (official path).
        global_pool=False is unsupported officially (models_vit.py:33 always deletes self.norm, causing AttributeError) — here it is a
        **toolkit alternative** that reuses the MAE-pretrained norm and outputs the cls token (assumption flag global_pool)."""
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            return self.fc_norm(x[:, 1:, :].mean(dim=1))
        return self.norm(x)[:, 0]


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# =================================================================================================== masking / patches
def random_masking(x: torch.Tensor, mask_ratio: float, noise: Optional[torch.Tensor] = None):
    """models_mae.random_masking (noise can be injected — for goldens and mask synchronization). mask_ratio 0 keeps everything."""
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    if noise is None:
        noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def patchify(imgs: torch.Tensor, p: int = 16) -> torch.Tensor:
    """(N, 1, T, F) → (N, (T/p)(F/p), p*p) — t-major (models_mae.patchify audio_exp)."""
    h, w = imgs.shape[2] // p, imgs.shape[3] // p
    x = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(imgs.shape[0], h * w, p * p)


def unpatchify(x: torch.Tensor, img_size=(1024, 128), p: int = 16) -> torch.Tensor:
    h, w = img_size[0] // p, img_size[1] // p
    x = x.reshape(x.shape[0], h, w, p, p, 1)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], 1, h * p, w * p)


def mae_recon_loss(imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, norm_pix_loss: bool = False, p: int = 16,
                   scope: str = "masked"):
    """models_mae.forward_loss: mean MSE over removed patches (scope='masked', official MAE).
    Paper Eq. 5 is written as MSE over the whole spectrogram, so restricting it to masked patches is an **assumption** (flag recon_scope; 'all' = mean over all patches)."""
    target = patchify(imgs, p)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5
    loss = ((pred - target) ** 2).mean(dim=-1)
    if scope == "all":
        return loss.mean()
    return (loss * mask).sum() / mask.sum()


# =================================================================================================== Swin decoder block (timm_patch)
def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws[0], ws[0], W // ws[1], ws[1], C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws[0], ws[1], C)


def _window_reverse(windows, ws, img_size):
    H, W = img_size
    B = int(windows.shape[0] / (H * W / ws[0] / ws[1]))
    x = windows.view(B, H // ws[0], W // ws[1], ws[0], ws[1], -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class WindowMultiHeadAttention(nn.Module):
    """AudioMAE timm_patch/swin_transformer.py WindowMultiHeadAttention: scaled-cosine attn(tau) + log-CPB meta MLP(2-384-heads, dropout 0.1)."""

    def __init__(self, dim, num_heads, window_size, drop_attn=0.0, drop_proj=0.0, meta_hidden_dim=384):
        super().__init__()
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        self.meta_mlp = _Mlp(2, meta_hidden_dim, num_heads, act_layer=nn.ReLU, drop=0.1)
        self.tau = nn.Parameter(torch.ones(num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(self.window_size[0]), torch.arange(self.window_size[1])],
                                            indexing="ij"), dim=0).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).reshape(-1, 2).float()
        self.register_buffer("relative_coordinates_log", torch.sign(rel) * torch.log(1.0 + rel.abs()), persistent=False)

    def _rel_bias(self):
        area = self.window_size[0] * self.window_size[1]
        bias = self.meta_mlp(self.relative_coordinates_log)
        return bias.transpose(1, 0).reshape(self.num_heads, area, area).unsqueeze(0)

    def forward(self, x, mask=None):
        Bw, L, C = x.shape
        qkv = self.qkv(x).view(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        denom = torch.norm(q, dim=-1, keepdim=True) @ torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1)
        attn = q @ k.transpose(-2, -1) / denom.clamp(min=1e-6)
        attn = attn / self.tau.clamp(min=0.01).reshape(1, self.num_heads, 1, 1)
        attn = attn + self._rel_bias()
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(Bw // nw, nw, self.num_heads, L, L) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(Bw, L, -1)
        return self.proj_drop(self.proj(x))


class SwinDecoderBlock(nn.Module):
    """AudioMAE timm_patch SwinTransformerBlock (post-norm, extra_norm=False, drop_path 0)."""

    def __init__(self, dim, num_heads, feat_size, window_size, shift_size=(0, 0), mlp_ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.feat_size = tuple(feat_size)
        target_shift = tuple(shift_size)
        self.window_size = tuple(f if f <= w else w for f, w in zip(self.feat_size, window_size))
        self.shift_size = tuple(0 if f <= w else s for f, w, s in zip(self.feat_size, self.window_size, target_shift))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn = WindowMultiHeadAttention(dim, num_heads, self.window_size)
        self.norm1 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), dim)
        self.norm2 = norm_layer(dim)
        self.register_buffer("attn_mask", self._make_attention_mask(), persistent=False)

    def _make_attention_mask(self):
        if not any(self.shift_size):
            return None
        H, W = self.feat_size
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        for h in (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None)):
            for w in (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = _window_partition(img_mask, self.window_size).view(-1, self.window_area)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        return am.masked_fill(am != 0, float(-100.0)).masked_fill(am == 0, float(0.0))

    def _shifted_window_attn(self, x):
        H, W = self.feat_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        sh, sw = self.shift_size
        if any(self.shift_size):
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
        xw = _window_partition(x, self.window_size).view(-1, self.window_area, C)
        aw = self.attn(xw, mask=self.attn_mask).view(-1, self.window_size[0], self.window_size[1], C)
        x = _window_reverse(aw, self.window_size, self.feat_size)
        if any(self.shift_size):
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
        return x.view(B, L, C)

    def forward(self, x):
        x = x + self.norm1(self._shifted_window_attn(x))
        x = x + self.norm2(self.mlp(x))
        return x


# =================================================================================================== decoder
class MAEDecoder(nn.Module):
    def __init__(self, in_dim=768, num_patches=512, patch_hw=(8, 64), decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mlp_ratio=4.0, patch_size=16, in_chans=1, decoder_mode=1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_mode = int(decoder_mode)
        self.in_dim = in_dim
        self.decoder_embed = nn.Linear(in_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        feat_size = (patch_hw[1], patch_hw[0])  # (64, 8)
        if self.decoder_mode == 1:
            self.decoder_blocks = nn.ModuleList([
                SwinDecoderBlock(decoder_embed_dim, 16, feat_size=feat_size, window_size=(4, 4),
                                 shift_size=(0, 0) if i % 2 == 0 else (2, 0), mlp_ratio=mlp_ratio, norm_layer=norm_layer)
                for i in range(16)])
        else:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(
            get_2d_sincos_pos_embed(decoder_embed_dim, patch_hw, cls_token=True)).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """x: (B, 1+keep, in_dim) [cls included] → pred (B, L, p*p)  (models_mae.forward_decoder)."""
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        if self.decoder_mode != 0:
            x = x[:, 1:, :]
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        if self.decoder_mode == 0:
            pred = pred[:, 1:, :]
        return pred


# =================================================================================================== ckpt loading
def load_audiomae_checkpoint(encoder: Optional[AudioMAEEncoder], path, decoder: Optional[MAEDecoder] = None,
                             skip_decoder_embed: bool = False) -> Dict:
    """
    Strict load of only name- and shape-matching keys from pretrained.pth['model'] into the encoder/decoder modules.
    - Encoder: fc_norm is not in the ckpt (new LN of models_vit) → excluded; all other keys are required.
    - Decoder: skip_decoder_embed=True skips decoder_embed.* (DDE concat 1536 input → assumed newly initialized).
    pos_embed / decoder_pos_embed computed before loading are verified to match the ckpt values (guards against the flatten-order quirk).
    """
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    info = {"path": str(path), "encoder_used": [], "decoder_used": [], "dropped": []}
    used = set()
    if encoder is not None:
        names = set(encoder.state_dict().keys())  # non-persistent buffers excluded
        kept = {k: v for k, v in state.items() if k in names and not k.startswith("fc_norm.")}
        missing = sorted(n for n in names if n not in kept and not n.startswith("fc_norm."))
        if missing:
            raise RuntimeError(f"Audio-MAE ckpt is missing encoder keys: {missing[:5]}")
        if not torch.allclose(encoder.pos_embed, kept["pos_embed"], atol=1e-6):
            raise RuntimeError("computed pos_embed differs from the ckpt (check the flatten-order quirk)")
        sd = encoder.state_dict()
        for k, v in kept.items():
            if tuple(sd[k].shape) != tuple(v.shape):
                raise RuntimeError(f"encoder shape mismatch {k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}")
        encoder.load_state_dict(kept, strict=False)
        info["encoder_used"] = sorted(kept)
        used |= set(kept)
    if decoder is not None:
        names = set(decoder.state_dict().keys())  # non-persistent buffers (attn_mask, relative_coordinates_log) excluded
        kept = {k: v for k, v in state.items() if k in names and not (skip_decoder_embed and k.startswith("decoder_embed."))}
        missing = sorted(n for n in names if n not in kept and not (skip_decoder_embed and n.startswith("decoder_embed.")))
        if missing:
            raise RuntimeError(f"Audio-MAE ckpt is missing decoder keys (check decoder_mode): {missing[:5]}")
        if not torch.allclose(decoder.decoder_pos_embed, kept["decoder_pos_embed"], atol=1e-6):
            raise RuntimeError("computed decoder_pos_embed differs from the ckpt")
        for k, v in kept.items():
            if tuple(decoder.state_dict()[k].shape) != tuple(v.shape):
                raise RuntimeError(f"decoder shape mismatch {k}: {tuple(v.shape)} vs {tuple(decoder.state_dict()[k].shape)}")
        decoder.load_state_dict(kept, strict=False)
        info["decoder_used"] = sorted(kept)
        used |= set(kept)
    info["dropped"] = sorted(k for k in state if k not in used)
    logger.info("Audio-MAE ckpt %s: encoder %d keys, decoder %d keys, dropped %d keys", path, len(info["encoder_used"]),
                len(info["decoder_used"]), len(info["dropped"]))
    return info


def _to_tf(x: torch.Tensor) -> torch.Tensor:
    """Toolkit fbank features (B, F=128, T=1024) [or (B,1,F,T)] → Audio-MAE input (B, 1, T, F)."""
    if x.dim() == 4:
        x = x.squeeze(1)
    return x.transpose(1, 2).unsqueeze(1).contiguous()


# =================================================================================================== classification model (B2 / stage 2)
@ModelFactory.register_model_class("audiomae_dde_cls")
class AudioMAEClassifier(BaseRSCModel, nn.Module):
    """
    Audio-MAE encoder + Linear classification (CE).
    - B2: fine-tune directly from the pretrained_path (AS-2M pretrained.pth) encoder.
    - Stage 2 (S2/S3): load only E_R (encoder_key prefix) from encoder_ckpt (stage-1 final_state.pth) — E_I, decoder, q_θ are discarded (assumption).
    Head token: global_pool=True → patch mean + fc_norm (Audio-MAE fine-tuning convention, assumption) / False → cls (flag).
    """

    def __init__(self, n_cls: int = 4, pretrained_path: Optional[str] = AUDIOMAE_CKPT, encoder_ckpt: Optional[str] = None,
                 encoder_key: str = "encoder_r.", global_pool: bool = True, img_size=(1024, 128), patch_size: int = 16,
                 head_init_std: float = 0.02):
        super().__init__()
        self.encoder = AudioMAEEncoder(img_size=tuple(img_size), patch_size=patch_size, global_pool=global_pool)
        self.load_info = None
        if encoder_ckpt is not None:
            self.load_info = self.load_stage1_encoder(encoder_ckpt, encoder_key)
        elif pretrained_path is not None:
            from rsc_toolkit.paths import resolve_pretrained

            self.load_info = load_audiomae_checkpoint(self.encoder, resolve_pretrained(explicit=pretrained_path))  # honors RSCTK_PRETRAINED_ROOT
        self.head = nn.Linear(self.encoder.embed_dim, n_cls)
        nn.init.trunc_normal_(self.head.weight, std=head_init_std)  # main_finetune_as.py:  trunc_normal_(head.weight, std=2e-5) is for AS; 0.02 assumed
        nn.init.zeros_(self.head.bias)

    def load_stage1_encoder(self, path, encoder_key: str = "encoder_r.") -> Dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        enc = {k[len(encoder_key):]: v for k, v in state.items() if k.startswith(encoder_key)}
        if not enc:
            raise RuntimeError(f"{path} has no '{encoder_key}' keys (check that it is a stage-1 final_state)")
        enc = {k: v for k, v in enc.items() if not k.startswith("fc_norm.")}  # stage 1 does not use fc_norm → keep the new LN
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        missing = [m for m in missing if not m.startswith("fc_norm.")]
        if missing or unexpected:
            raise RuntimeError(f"E_R load mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        others = sorted({k.split(".")[0] for k in state if not k.startswith(encoder_key)})
        logger.info("Loaded stage-1 E_R %s (%d keys); discarded: %s", path, len(enc), others)
        return {"loaded": sorted(enc), "discarded_prefixes": others}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.forward_features(_to_tf(x)))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)


# =================================================================================================== DDE-MAE (stage 1)
@ModelFactory.register_model_class("audiomae_dde")
class DDEMAE(BaseRSCModel, nn.Module):
    """
    variant='s3': E_R + E_I (original and shuffled in parallel), concat (1536) → decoder, L = L_recon + α1 L_siamese + α2 L_MI (Eq. 7).
    variant='s2': single encoder (E_R) + original decoder (768 input, pretrained decoder_embed loaded), L = L_recon.
    training_forward returns (dummy zero logits (B, n_cls), loss) — stage 1 has no evaluation (limit_val_batches 0).
    """

    def __init__(self, variant: str = "s3", n_cls: int = 4, pretrained_path: Optional[str] = AUDIOMAE_CKPT,
                 mask_ratio: float = 0.8, norm_pix_loss: bool = False, decoder_mode: int = 1,
                 alpha_siamese: float = 0.5, alpha_mi: float = 0.02, use_siamese: bool = True, use_mi: bool = True,
                 siamese_margin: float = 1.0, siamese_level: str = "pooled", mi_level: str = "pooled",
                 shuffle_unit: str = "patch", shuffle_mask: str = "same_physical",
                 club_hidden: int = 512, club_lr: float = 1e-4, club_steps: int = 1, recon_scope: str = "masked",
                 img_size=(1024, 128), patch_size: int = 16):
        super().__init__()
        if variant not in ("s2", "s3"):
            raise ValueError(f"variant must be 's2'|'s3', got {variant!r}")
        if siamese_level not in ("pooled", "token") or mi_level not in ("pooled",):
            raise ValueError("siamese_level ∈ {pooled, token}, mi_level ∈ {pooled}")
        if shuffle_mask not in ("same_physical", "same_index"):
            raise ValueError("shuffle_mask ∈ {same_physical, same_index}")
        if recon_scope not in ("masked", "all"):
            raise ValueError("recon_scope ∈ {masked, all}")
        self.recon_scope = recon_scope
        if shuffle_unit == "frame" and shuffle_mask == "same_physical":
            raise ValueError("shuffle_unit=frame cannot synchronize physical patches → use shuffle_mask=same_index")
        self.variant, self.n_cls = variant, n_cls
        self.mask_ratio, self.norm_pix_loss, self.patch_size = float(mask_ratio), bool(norm_pix_loss), int(patch_size)
        self.alpha_siamese, self.alpha_mi = float(alpha_siamese), float(alpha_mi)
        self.use_siamese, self.use_mi = bool(use_siamese) and variant == "s3", bool(use_mi) and variant == "s3"
        self.siamese_margin, self.siamese_level, self.mi_level = float(siamese_margin), siamese_level, mi_level
        self.shuffle_unit, self.shuffle_mask = shuffle_unit, shuffle_mask
        self.club_steps = int(club_steps)

        self.encoder_r = AudioMAEEncoder(img_size=tuple(img_size), patch_size=patch_size, global_pool=False)
        D = self.encoder_r.embed_dim
        dual = variant == "s3"
        self.decoder = MAEDecoder(in_dim=2 * D if dual else D, num_patches=self.encoder_r.patch_embed.num_patches,
                                  patch_hw=self.encoder_r.patch_hw, decoder_mode=decoder_mode, patch_size=patch_size)
        self.load_info = None
        if pretrained_path is not None:
            # relative paths are searched in cwd, then under RSCTK_PRETRAINED_ROOT
            self.load_info = load_audiomae_checkpoint(self.encoder_r, str(resolve_pretrained(explicit=pretrained_path)),
                                                      self.decoder, skip_decoder_embed=dual)
        # E_I = duplicate of the pretrained encoder (paper §2.3 "duplicate the pretrained encoder")
        self.encoder_i = copy.deepcopy(self.encoder_r) if dual else None
        self.club = VCLUB(D, D, club_hidden) if self.use_mi else None
        self._club_opt = torch.optim.Adam(self.club.parameters(), lr=float(club_lr)) if self.use_mi else None
        self.last_losses: Dict[str, float] = {}

    # ---------------------------------------------------------------- components
    @staticmethod
    def _pool(latent: torch.Tensor) -> torch.Tensor:
        return latent[:, 1:, :].mean(dim=1)  # mean of patch tokens excluding cls (assumption)

    def forward_stage1(self, x_tf: torch.Tensor, noise: Optional[torch.Tensor] = None, perm: Optional[torch.Tensor] = None):
        """
        x_tf: (B, 1, T, F). Returns dict(loss, recon, siamese, mi, pred, mask, z_r, z_i, z_i_shuf).
        noise/perm are injected values for reproducible tests.
        """
        B = x_tf.shape[0]
        L = self.encoder_r.patch_embed.num_patches
        if noise is None:
            noise = torch.rand(B, L, device=x_tf.device)
        lat_r, mask, ids_restore = self.encoder_r.forward_tokens(x_tf, self.mask_ratio, noise)
        out = {"mask": mask, "ids_restore": ids_restore}
        if self.variant == "s2":
            pred = self.decoder(lat_r, ids_restore)
            recon = mae_recon_loss(x_tf, pred, mask, self.norm_pix_loss, self.patch_size, self.recon_scope)
            out.update(pred=pred, recon=recon, loss=recon, z_r=lat_r)
            return out
        lat_i, mask_i, ids_restore_i = self.encoder_i.forward_tokens(x_tf, self.mask_ratio, noise)  # same mask (assumption)
        pred = self.decoder(torch.cat([lat_r, lat_i], dim=-1), ids_restore)  # per-token feature concat (Fig. 1)
        recon = mae_recon_loss(x_tf, pred, mask, self.norm_pix_loss, self.patch_size, self.recon_scope)
        loss = recon
        out.update(pred=pred, recon=recon, z_r=lat_r, z_i=lat_i)
        if self.use_siamese:
            if perm is None:
                x_shuf, perm = time_shuffle(x_tf, self.shuffle_unit, self.patch_size)
            else:
                x_shuf = _apply_perm(x_tf, perm, self.shuffle_unit, self.patch_size)
            if self.shuffle_unit == "patch" and self.shuffle_mask == "same_physical":
                noise_s = permute_token_noise(noise, perm, self.encoder_r.grid_t, self.encoder_r.grid_f)
            else:
                noise_s = noise
            lat_i_s, _, _ = self.encoder_i.forward_tokens(x_shuf, self.mask_ratio, noise_s)
            if self.siamese_level == "pooled":
                sia = siamese_hinge(self._pool(lat_i), self._pool(lat_i_s), self.siamese_margin)
            else:
                # token level: with same_physical, permute_token_noise preserves even the argsort ranks, so the k-th kept token is
                # the same physical patch in both inputs — tokens correspond directly. With same_index, same grid positions are compared.
                sia = siamese_hinge(lat_i[:, 1:], lat_i_s[:, 1:], self.siamese_margin)
            loss = loss + self.alpha_siamese * sia
            out.update(siamese=sia, z_i_shuf=lat_i_s, perm=perm)
        if self.use_mi:
            zr, zi = self._pool(lat_r), self._pool(lat_i)
            if self.training and torch.is_grad_enabled():
                self._club_step(zr.detach(), zi.detach())
            mi = self.club.mi_upper_bound_detached_q(zr, zi)
            loss = loss + self.alpha_mi * mi
            out.update(mi=mi)
        out["loss"] = loss
        return out

    def _club_step(self, zr: torch.Tensor, zi: torch.Tensor):
        """Auxiliary q_θ update: maximize log q_θ(z_I|z_R) (inputs detached — main parameters unchanged)."""
        for _ in range(self.club_steps):
            self._club_opt.zero_grad(set_to_none=True)
            nll = -self.club.loglikelihood(zr, zi)
            nll.backward()
            self._club_opt.step()
        self._club_opt.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- training contract

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        x_tf = _to_tf(audio_batch)
        out = self.forward_stage1(x_tf)
        self.last_losses = {k: float(out[k].detach()) for k in ("recon", "siamese", "mi") if k in out}
        dummy = torch.zeros(x_tf.shape[0], self.n_cls, device=x_tf.device)
        return dummy, out["loss"]

    def forward(self, x):
        return self.forward_stage1(_to_tf(x))["pred"]


def _apply_perm(x_tf: torch.Tensor, perm: torch.Tensor, unit: str, patch_frames: int) -> torch.Tensor:
    B, C, T, Fd = x_tf.shape
    if unit == "patch":
        n = T // patch_frames
        xs = x_tf.reshape(B, C, n, patch_frames, Fd)
        return torch.gather(xs, 2, perm.view(B, 1, n, 1, 1).expand(B, C, n, patch_frames, Fd)).reshape(B, C, T, Fd)
    return torch.gather(x_tf, 2, perm.view(B, 1, T, 1).expand(B, C, T, Fd))
