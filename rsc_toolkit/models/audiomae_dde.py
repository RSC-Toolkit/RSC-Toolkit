"""
Audio-MAE 백본 + DDE-MAE(Disentangling Dual-Encoder MAE) — Wei·Miao·Li, Interspeech 2025 (arXiv 2506.10698).

[P0 골든 — 공식 코드 있음] Audio-MAE(facebookresearch/AudioMAE, models_mae.py / models_vit.py, timm 0.3.2 계보) 를 timm 1.0 Block 위에
재구현. 골든(scripts/reproduction/dump_ddemae_goldens.py, 별도 venv CPU) 과 1e-4 일치:
  - AudioMAEEncoder : ViT-B/16, img_size (1024, 128)=(T, F), patch 16 → grid 64x8=512 토큰(t-major), cls_token,
                      고정 2D sin-cos pos_embed (공식 get_2d_sincos_pos_embed_flexible(patch_hw=(8,64)) — 격자 (8,64) 로 계산해 t-major
                      토큰에 붙이는 공식 quirk 그대로; ckpt 값과 일치 검증), timm Block x12 (768, heads 12, mlp 4, qkv_bias, LN eps 1e-6),
                      norm (MAE latent) / fc_norm(global_pool 파인튜닝, models_vit.py:52-54).
  - MAEDecoder      : decoder_embed Linear(in_dim → 512), mask_token, decoder_pos_embed(513, 512) 고정 sin-cos,
                      decoder_mode=1 (공개 AS-2M ckpt 가 이걸로 학습됨): AudioMAE timm_patch/swin_transformer.py SwinTransformerBlock x16
                      (dim 512, heads 16, feat (64,8), window (4,4), 홀수 블록 shift (2,0), scaled-cosine attn + log-CPB meta MLP, post-norm),
                      cls 제거 후 (64,8) 격자 처리; decoder_mode=0: timm Block x8 (전역 어텐션, cls 유지 후 제거). decoder_norm, decoder_pred(512→256).
  - random_masking / patchify / unpatchify / MAE 재구성 손실 (models_mae.py:225-247, :171-197, :395-411).
  - load_audiomae_checkpoint: pretrained.pth['model'] 에서 이름·모양 일치 키만 strict 로드 (인코더 150 키, 디코더 280 키 = 430; drop 0).

[논문 명세 재구현 — 코드 없음, 골든 없음] DDE 부분 (rsc_toolkit/methods/dde_mae.py 와 함께):
  - DDEMAE(model 'audiomae_dde', 1단계): 사전학습 인코더 복제 → E_R(원본 mel), E_I(원본 mel + time-shuffle mel 병렬).
      결합 = 토큰별 특징 차원 concat (Fig.1 판독; 1536 → decoder_embed 신규 초기화 [가정]) → 디코더 → 마스크 mel 재구성 L_recon(식 5 MSE).
      L_siamese(식 6, margin 가정 1.0, 풀링 벡터 가정), L_MI = vCLUB(식 4, 풀링 벡터 가정, q_θ 보조 Adam lr 1e-4 가정 매 스텝 1회).
      총손실 식 7: L = L_recon + 0.5 L_siamese + 0.02 L_MI. variant='s2' 는 단일 인코더 + 원 디코더 재구성(L_recon)만.
  - AudioMAEClassifier(model 'audiomae_dde_cls', B2/2단계): 인코더(+ encoder_ckpt 로 1단계 E_R 만 로드) → global_pool+fc_norm [가정] → Linear → CE.

가정 목록(전부 config 플래그, docs/AUTHOR_QUESTIONS_2026-08-26.md DDE-MAE 절): mask_ratio 0.8, norm_pix_loss False, decoder_mode 1(ckpt 와 키 일치),
decoder_embed 1536→512 신규 초기화, time-shuffle 단위 'patch', E_I 두 입력 마스크 = 물리적으로 같은 패치, Siamese/MI 풀링 벡터, margin 1.0,
q_θ MLP(768-512-768) + 보조 Adam 1e-4, 분류 헤드 global_pool(fc_norm), E_I 폐기.

quirk: pl_adapter 가 단일 optimizer 라 q_θ 는 모델 내부 보조 optimizer 로 갱신한다(training_forward 안). 주 손실의 MI 항은 q_θ 파라미터를
detach 한 함수형 호출로 계산해 주 optimizer 가 q_θ 를 건드리지 않게 한다 (grad None → AdamW 스킵). 보조 optimizer 상태는 last.ckpt 에 없다.
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


# =================================================================================================== 인코더
class AudioMAEEncoder(nn.Module):
    """공식 MaskedAutoencoderViT 인코더부(models_mae.py) + models_vit.py 의 global_pool/fc_norm."""

    def __init__(self, img_size=(1024, 128), patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, global_pool: bool = True):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        ps = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.patch_embed = PatchEmbed(tuple(img_size), ps, in_chans, embed_dim)
        gt, gf = self.patch_embed.grid_size  # (T/16, F/16) = (64, 8)
        self.patch_hw = (gf, gt)  # 공식 PatchEmbed_org.patch_hw = (img_size[1]//p, img_size[0]//p) = (8, 64)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.global_pool = bool(global_pool)
        self.fc_norm = norm_layer(embed_dim)  # models_vit.py: global_pool=True 일 때만 쓰는 새 LN (ckpt 에 없음)
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
        반환 (latent (B, 1+keep, D), mask (B, L) [1=제거], ids_restore (B, L)).
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
        """models_vit.VisionTransformer.forward_features: global_pool → mean(패치) → fc_norm (공식 경로).
        global_pool=False 는 공식 미지원(models_vit.py:33 이 self.norm 을 무조건 삭제해 AttributeError) — 여기서는 MAE 사전학습 norm 을
        재사용해 cls 토큰을 내는 **자체 대안**(가정 플래그 global_pool)."""
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


# =================================================================================================== 마스킹·패치
def random_masking(x: torch.Tensor, mask_ratio: float, noise: Optional[torch.Tensor] = None):
    """models_mae.random_masking (noise 주입 가능 — 골든·마스크 동기화용). mask_ratio 0 이면 전체 유지."""
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
    """models_mae.forward_loss: 제거된 패치의 MSE 평균 (scope='masked', 공식 MAE).
    논문 식 5 는 전체 스펙트로그램 MSE 로 쓰여 있어 마스크 한정은 **가정**(플래그 recon_scope; 'all' = 전 패치 평균)."""
    target = patchify(imgs, p)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5
    loss = ((pred - target) ** 2).mean(dim=-1)
    if scope == "all":
        return loss.mean()
    return (loss * mask).sum() / mask.sum()


# =================================================================================================== Swin 디코더 블록 (timm_patch)
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


# =================================================================================================== 디코더
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
        """x: (B, 1+keep, in_dim) [cls 포함] → pred (B, L, p*p)  (models_mae.forward_decoder)."""
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


# =================================================================================================== ckpt 로드
def load_audiomae_checkpoint(encoder: Optional[AudioMAEEncoder], path, decoder: Optional[MAEDecoder] = None,
                             skip_decoder_embed: bool = False) -> Dict:
    """
    pretrained.pth['model'] 을 인코더/디코더 모듈에 이름·모양 일치 키만 strict 로드.
    - 인코더: fc_norm 은 ckpt 에 없다(models_vit 의 새 LN) → 제외하고 나머지 전부 필수.
    - 디코더: skip_decoder_embed=True 면 decoder_embed.* 를 건너뛴다 (DDE concat 1536 입력 → 신규 초기화 가정).
    pos_embed / decoder_pos_embed 는 로드 전 계산값과 ckpt 값이 일치하는지 검증(flatten 순서 quirk 방어).
    """
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    info = {"path": str(path), "encoder_used": [], "decoder_used": [], "dropped": []}
    used = set()
    if encoder is not None:
        names = set(encoder.state_dict().keys())  # 비영속 버퍼 제외
        kept = {k: v for k, v in state.items() if k in names and not k.startswith("fc_norm.")}
        missing = sorted(n for n in names if n not in kept and not n.startswith("fc_norm."))
        if missing:
            raise RuntimeError(f"Audio-MAE ckpt 에 인코더 키가 없다: {missing[:5]}")
        if not torch.allclose(encoder.pos_embed, kept["pos_embed"], atol=1e-6):
            raise RuntimeError("pos_embed 계산값이 ckpt 와 다르다 (flatten 순서 quirk 확인)")
        sd = encoder.state_dict()
        for k, v in kept.items():
            if tuple(sd[k].shape) != tuple(v.shape):
                raise RuntimeError(f"인코더 모양 불일치 {k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}")
        encoder.load_state_dict(kept, strict=False)
        info["encoder_used"] = sorted(kept)
        used |= set(kept)
    if decoder is not None:
        names = set(decoder.state_dict().keys())  # 비영속 버퍼(attn_mask, relative_coordinates_log) 제외
        kept = {k: v for k, v in state.items() if k in names and not (skip_decoder_embed and k.startswith("decoder_embed."))}
        missing = sorted(n for n in names if n not in kept and not (skip_decoder_embed and n.startswith("decoder_embed.")))
        if missing:
            raise RuntimeError(f"Audio-MAE ckpt 에 디코더 키가 없다 (decoder_mode 확인): {missing[:5]}")
        if not torch.allclose(decoder.decoder_pos_embed, kept["decoder_pos_embed"], atol=1e-6):
            raise RuntimeError("decoder_pos_embed 계산값이 ckpt 와 다르다")
        for k, v in kept.items():
            if tuple(decoder.state_dict()[k].shape) != tuple(v.shape):
                raise RuntimeError(f"디코더 모양 불일치 {k}: {tuple(v.shape)} vs {tuple(decoder.state_dict()[k].shape)}")
        decoder.load_state_dict(kept, strict=False)
        info["decoder_used"] = sorted(kept)
        used |= set(kept)
    info["dropped"] = sorted(k for k in state if k not in used)
    logger.info("Audio-MAE ckpt %s: 인코더 %d 키, 디코더 %d 키, drop %d 키", path, len(info["encoder_used"]),
                len(info["decoder_used"]), len(info["dropped"]))
    return info


def _to_tf(x: torch.Tensor) -> torch.Tensor:
    """툴킷 fbank 특징 (B, F=128, T=1024) [또는 (B,1,F,T)] → Audio-MAE 입력 (B, 1, T, F)."""
    if x.dim() == 4:
        x = x.squeeze(1)
    return x.transpose(1, 2).unsqueeze(1).contiguous()


# =================================================================================================== 분류 모델 (B2 / 2단계)
@ModelFactory.register_model_class("audiomae_dde_cls")
class AudioMAEClassifier(BaseRSCModel, nn.Module):
    """
    Audio-MAE 인코더 + Linear 분류 (CE).
    - B2: pretrained_path(AS-2M pretrained.pth) 인코더에서 바로 미세조정.
    - 2단계(S2/S3): encoder_ckpt(1단계 final_state.pth) 의 E_R(encoder_key 접두)만 로드 — E_I·디코더·q_θ 는 폐기(가정).
    헤드 토큰: global_pool=True → 패치 평균 + fc_norm(Audio-MAE 파인튜닝 관례, 가정) / False → cls (플래그).
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

            self.load_info = load_audiomae_checkpoint(self.encoder, resolve_pretrained(explicit=pretrained_path))  # RSCTK_PRETRAINED_ROOT 인정
        self.head = nn.Linear(self.encoder.embed_dim, n_cls)
        nn.init.trunc_normal_(self.head.weight, std=head_init_std)  # main_finetune_as.py:  trunc_normal_(head.weight, std=2e-5) 는 AS 용; 가정 0.02
        nn.init.zeros_(self.head.bias)

    def load_stage1_encoder(self, path, encoder_key: str = "encoder_r.") -> Dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        enc = {k[len(encoder_key):]: v for k, v in state.items() if k.startswith(encoder_key)}
        if not enc:
            raise RuntimeError(f"{path} 에 '{encoder_key}' 키가 없다 (1단계 final_state 인지 확인)")
        enc = {k: v for k, v in enc.items() if not k.startswith("fc_norm.")}  # 1단계는 fc_norm 을 쓰지 않았다 → 새 LN 유지
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        missing = [m for m in missing if not m.startswith("fc_norm.")]
        if missing or unexpected:
            raise RuntimeError(f"E_R 로드 불일치: missing {missing[:5]} unexpected {unexpected[:5]}")
        others = sorted({k.split(".")[0] for k in state if not k.startswith(encoder_key)})
        logger.info("1단계 E_R 로드 %s (%d 키); 폐기: %s", path, len(enc), others)
        return {"loaded": sorted(enc), "discarded_prefixes": others}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.forward_features(_to_tf(x)))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)


# =================================================================================================== DDE-MAE (1단계)
@ModelFactory.register_model_class("audiomae_dde")
class DDEMAE(BaseRSCModel, nn.Module):
    """
    variant='s3': E_R + E_I(원본·셔플 병렬), concat(1536) → 디코더, L = L_recon + α1 L_siamese + α2 L_MI (식 7).
    variant='s2': 단일 인코더(E_R) + 원 디코더(768 입력, 사전학습 decoder_embed 로드), L = L_recon.
    training_forward 는 (더미 logits (B, n_cls) 0, loss) 를 돌려준다 — 1단계는 평가가 없다 (limit_val_batches 0).
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
            raise ValueError("shuffle_unit=frame 은 패치 단위 물리 동기화가 불가 → shuffle_mask=same_index 를 써라")
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
            # 상대 경로는 cwd → RSCTK_PRETRAINED_ROOT 아래 순으로 찾는다 (P2-9 경로 독립; 예전엔 직행)
            self.load_info = load_audiomae_checkpoint(self.encoder_r, str(resolve_pretrained(explicit=pretrained_path)),
                                                      self.decoder, skip_decoder_embed=dual)
        # E_I = 사전학습 인코더 복제 (논문 §2.3 "duplicate the pretrained encoder")
        self.encoder_i = copy.deepcopy(self.encoder_r) if dual else None
        self.club = VCLUB(D, D, club_hidden) if self.use_mi else None
        self._club_opt = torch.optim.Adam(self.club.parameters(), lr=float(club_lr)) if self.use_mi else None
        self.last_losses: Dict[str, float] = {}

    # ---------------------------------------------------------------- 부품
    @staticmethod
    def _pool(latent: torch.Tensor) -> torch.Tensor:
        return latent[:, 1:, :].mean(dim=1)  # cls 제외 패치 토큰 평균 (가정)

    def forward_stage1(self, x_tf: torch.Tensor, noise: Optional[torch.Tensor] = None, perm: Optional[torch.Tensor] = None):
        """
        x_tf: (B, 1, T, F). 반환 dict(loss, recon, siamese, mi, pred, mask, z_r, z_i, z_i_shuf).
        noise/perm 은 테스트 재현용 주입값.
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
        lat_i, mask_i, ids_restore_i = self.encoder_i.forward_tokens(x_tf, self.mask_ratio, noise)  # 같은 마스크(가정)
        pred = self.decoder(torch.cat([lat_r, lat_i], dim=-1), ids_restore)  # 토큰별 특징 concat (Fig.1)
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
                # token 단위: same_physical 이면 permute_token_noise 가 argsort 순위까지 보존하므로 k 번째 유지 토큰이
                # 두 입력에서 같은 물리 패치 — 토큰별로 바로 대응. same_index 면 같은 격자 위치끼리 비교.
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
        """q_θ 보조 갱신: log q_θ(z_I|z_R) 최대화 (입력은 detach — 주 파라미터 불변)."""
        for _ in range(self.club_steps):
            self._club_opt.zero_grad(set_to_none=True)
            nll = -self.club.loglikelihood(zr, zi)
            nll.backward()
            self._club_opt.step()
        self._club_opt.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- 학습 계약
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
