"""
LungAdapter(Interspeech 2024, Xiao et al.) — 동결 AST + Attention-Adapter/Res-Adapter, **논문 명세 재구현** (공식·비공식 코드 없음).

논문: LungAdapter: Efficient Adapting Audio Spectrogram Transformer for Lung Sound Classification
      (DOI 10.21437/Interspeech.2024-106; docs/paper_recon_2026-08-26.json['lungadapter'] 정찰·verdict 기준)

골격: 논문 'experimental setup mostly follows [13]'(Bae Patch-Mix) + 'AST pre-trained from Bae et al. on ImageNet and Audioset'
      → 툴킷 ast_patchmix.PatchMixAST(IN+AS ckpt audioset_0.4593.pth, 뒤집힌 798×128 quirk 포함)를 그대로 부모로 쓴다.
      백본(self.v.*)·mlp_head 는 전부 동결, 어댑터 + classifier(LN+Linear, 식4)만 학습.

각 timm Block(12개)에 (논문 Fig.1·식 5-8):
  (a) Attention-Adapter — MHSA 출력 뒤 직렬 ('positioned after the MHSA layer'):
        h  = W_down(s1·X)                       [768→16]
        h' = h + (1/3)·Σ_{k=3,5,7} DWConv_k(h)
        y  = W_up[ s2·GELU(Conv_1x1(h') + h') ] + s1·X
      원형은 참고문헌 [22] Mona(MonaOp: 2D DWConv 3/5/7 평균+identity → 1×1 conv+identity → GELU → up-proj).
  (b) Res-Adapter — MLP 와 병렬(AdaptFormer 류): W_up(GELU(W_down(X))), bottleneck ratio 0.15 → round(768·0.15)=115.
      Fig.1 중앙 패널(PDF 벡터 기하)에서 Res-Adapter 입력선은 MHSA 잔차합 ⊕ **뒤**·두 번째 LayerNorm **앞**의 잔차 스트림에서
      분기하고 출력은 MLP 뒤 ⊕ 로 들어간다 → 입력 = x (norm2(x) 아님). 그림은 MLP 의 항등 skip 을 그리지 않지만
      식(3)·동결 AST 유지로 해석해 x + MLP(norm2 x) + ResAdapter(x) 로 구현한다.

논문이 확정하지 않은 항목은 전부 플래그이며 기본값은 파라미터 수(Table 1/2: Attn만 0.34M · Res만 2.14M · 둘 다 2.48M, ±0.01M)에
동시에 맞는 조합 **중 하나**다 (tests/unit/models/test_ast_lungadapter.py 에 조합별 표 고정). 정찰 verdict.corrections 가 지적하듯
Res 115 bias 없음+LN(2,142,724) / Attn 1D+LN+벡터 s1(343,300) 등 다른 조합도 세 값을 ±0.01M 로 동시에 만족하므로
파라미터 수만으로 1D/2D·스케일 형태·LN/bias 유무를 확정할 수 없다 (저자 확인 필요):
  conv_dim      : "2d"(기본; 패치 격자 freq 12 × time 79, cls/dist 토큰은 DWConv 우회) | "1d"(토큰 축, cls/dist 포함)
  scale_form    : "vector"(기본; s1∈R^768, s2∈R^16 학습) | "scalar"(s1, s2 스칼라 학습)
  attn_insert   : "pre_residual"(기본; x = x + Adapter(attn(norm1 x))) | "post_residual"(Mona 코드: x = x + attn(...); x = Adapter(x))
  res_input     : "x"(기본; Fig.1 분기점 = ⊕ 뒤·LN2 앞 잔차 스트림) | "norm2"(MLP 와 같은 norm2(x) 입력 — 대안 플래그)
  res_scale     : "none"(기본; 논문 식(8)·Fig.1 우측(FC-GELU-FC, 좌측과 달리 S 기호 없음) 그대로, 상수 res_scale_init=1)
                  | "vector"(768-d 학습 스케일) | "scalar"(학습 스칼라)
                  ※ "vector" 는 Res-only 2,134,888(2.13M)·둘 다 2,468,392(2.47M)을 Table 1/2 의 2.14M/2.48M 에 맞추는
                    파라미터 맞춤 추정이라 기본값으로 굳히지 않는다. 기본 조합은 Attn만 0.34 ✓ / Res만 2.13(2.14 와 0.0051 차, ±0.01M 안)
                    / 둘 다 2.468(2.48 과 0.0116 차, ±0.01M 을 0.0016 초과) — 논문 카운트가 블록당 ≈770 개(총 ≈9k) 더 많은 이유는
                    미해결(저자 질문). 세 값을 동시에 맞추려면 model.params.res_scale=vector.
  zero_init_up  : True(기본; 두 어댑터의 W_up 을 0 으로 초기화 → 학습 시작 시 동결 AST 와 동일 출력, AdaptFormer/LoRA 관례)
  s1/s2 초기값 1.0, res_scale 초기값 1.0 (Mona gammax=1 / 논문 미기재 — 추정)
  어댑터 dropout 없음 (Mona 원형의 dropout 0.1 미채택; 논문 미기재 — 추정)
  weight decay(옵티마이저 wd 1e-6, Bae 상속 추정)는 s1/s2 스케일 파라미터에도 걸린다 — 논문 근거 없음, 영향 미미(플래그 없음)
기본 조합(2D·벡터 s1/s2·res_scale none)의 학습 파라미터: Attn만 338,116 / Res만 2,134,888 / 둘 다 2,468,392
(Total = PatchMixAST 87,531,736 + 어댑터 = 87,865,240 / 89,662,012 / 89,995,516) — 논문 0.34M/2.14M/2.48M
(Total 87.87/89.67/90.01M 은 87.53+Trainable 반올림 합) 과 비교하면 '둘 다' 만 0.0116M 부족(위 res_scale 항목).
res_scale=vector 이면 338,116 / 2,144,104 / 2,477,608 (Total 90,004,732) 로 세 값 모두 ±0.01M 안.

전처리 표기: 논문 3.1.2 '25 ms Hamming window' vs 상속한 icbhi_patchmix(kaldi fbank window_type='hanning', Bae util/icbhi_util.py:395
과 동일) — Bae 코드 상속으로 hanning 을 쓴다 (논문 표기와 불일치, 정찰 verdict 지적).

학습 레시피(논문 명시): Adam lr 5e-4, cosine, bs 8, 50 epoch, weighted CE. 나머지(SpecAugment icbhi_ast_sup, MA 0.5, AMP, wd 1e-6,
best-on-test Score 최고 && Se>5)는 Bae 상속 **추정** → configs/lungadapter_icbhi.yaml·callbacks/lungadapter_paper.yaml 에서 조정.
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
    """식(6)-(7) 의 conv 부분 = Mona MonaOp: (1/3)Σ DWConv_{3,5,7} + identity → Conv1×1 + identity.

    conv_dim="2d": 입력 (B, C, H, W) 격자 / "1d": 입력 (B, C, N) 토큰 열. 반환은 GELU 이전 값 (Conv(h')+h')."""

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
        h_prime = h + sum(conv(h) for conv in self.dwconvs) / float(len(self.dwconvs))   # 식(6)
        return self.projector(h_prime) + h_prime                                          # 식(7) 괄호 안


class AttentionAdapter(nn.Module):
    """논문 식(5)-(7). forward(x, grid): x (B, N, D); grid=(H, W) 는 2D 일 때 패치 격자 (N = n_prefix + H·W)."""

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
            # 토큰 축 1D DWConv — cls/dist 토큰도 열에 포함(시퀀스 앞 2개)
            return self.conv(h.transpose(1, 2)).transpose(1, 2)
        if grid is None:
            raise ValueError("conv_dim='2d' 는 패치 격자 grid=(H, W) 가 필요하다")
        hh, ww = grid
        p = self.n_prefix
        if n != p + hh * ww:
            raise ValueError(f"토큰 수 {n} ≠ prefix {p} + {hh}×{ww}")
        patches = h[:, p:, :].reshape(b, hh, ww, c).permute(0, 3, 1, 2)
        patches = self.conv(patches).permute(0, 2, 3, 1).reshape(b, hh * ww, c)
        if p == 0:
            return patches
        # cls/dist 토큰: 격자에 없으므로 DWConv 항은 0 (h'=h), 1×1 conv(토큰별 선형)만 적용
        prefix = h[:, :p, :]
        prefix = self.conv.projector(prefix.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2) + prefix
        return torch.cat([prefix, patches], dim=1)

    def forward(self, x: torch.Tensor, grid: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        sx = self.s1 * x                                   # s1·X
        h = self.down(sx)                                  # 식(5)
        z = self._conv_tokens(h, grid)                     # Conv(h') + h'
        return self.up(self.s2 * F.gelu(z)) + sx           # 식(7)


class ResAdapter(nn.Module):
    """논문 식(8): W_up(GELU(W_down(X))). bottleneck = round(dim · ratio) (768·0.15 = 115.2 → 115)."""

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
    """timm Block 래퍼. 원 Block 의 norm1/attn/ls1/drop_path1/norm2/mlp/ls2/drop_path2 만 호출한다(버전 의존 최소화).

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
        self.grid: Optional[Tuple[int, int]] = None   # forward_features 가 매 호출 설정 (H=freq 패치, W=time 패치)

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
    """PatchMixAST(IN+AS, 뒤집힌 798×128 quirk) + 블록별 LungAdapter, 백본 동결. method 는 'ce' 고정."""

    def __init__(
        self,
        n_cls: int = 4,
        # ---- 어댑터 (논문 명시) ----
        use_attn_adapter: bool = True,
        use_res_adapter: bool = True,
        attn_hidden: int = 16,
        res_ratio: float = 0.15,
        res_bottleneck: Optional[int] = None,
        # ---- 논문 미기재 → 플래그 (모듈 docstring) ----
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
        # ---- 부모(PatchMixAST) 인자 ----
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
            logger.warning("LungAdapterAST: 어댑터 없음 — 동결 AST + 헤드(linear probe)만 학습한다")
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

        # 실제 패치 격자: forward_features 가 transpose(2,3) 뒤 conv 하므로 (H=freq 패치, W=time 패치) = (12, 79)
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)   # (79, 12) — quirk 인자 기준
        self.token_grid = (t_dim, f_dim)
        self._set_grid(self.token_grid)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze()
        tr, tot = self.count_parameters()
        logger.info("LungAdapterAST: trainable %.3fM / total %.3fM (%s)", tr / 1e6, tot / 1e6, self.adapter_cfg)

    # ------------------------------------------------------------------ 유틸
    def _set_grid(self, grid: Tuple[int, int]) -> None:
        for blk in self.v.blocks:
            blk.grid = grid

    def _freeze(self) -> None:
        """self.v.* (원 Block 포함) 과 mlp_head 동결; 어댑터·스케일·classifier 만 학습."""
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
        # 부모와 동일한 격자 계산 (transpose 전 x: (B,1,T,F) → 격자 H=(F-16)/10+1, W=(T-16)/10+1)
        grid = (int((x.size(3) - 16) / 10) + 1, int((x.size(2) - 16) / 10) + 1)
        self._set_grid(grid)
        return super().forward_features(x, y=y, patch_mix=patch_mix, time_domain=time_domain)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        return super().training_forward(audio_batch, labels, loss_func)
