"""
OPERA-CT 분류 모델 — NTT EMBC'25(arXiv 2504.18004) 동결 평가 경로(eval-audio-repr app/icbhi_sprs/solve.py + evar/ar_opera.py) 재현.

TaskNetwork(solve.py): ar.encode_frames(파형 배치) → [B, 768, 1] → transpose → [B, 1, 768] → 헤드.
  - AR_OPERA_CT.encode_frames: get_entire_signal_librosa(배치 trim·mel64·dB·배치 min-max) → augment_if_training(evar SpecAugment:
    torchaudio FrequencyMasking(20)·TimeMasking(40), iid_masks=False → **배치가 마스크 1벌을 공유**) → (B,T,64) → Cola.extract_feature(768)
    = HTSAT latent_output.
  - 헤드 'tfm' = TfmEncHead(sem 토큰 + timm Block×4, heads 1, mlp_ratio 1, xavier 초기화 뒤 Linear 는 기본 초기화 — 원본 순서 유지),
    'mlp' = MLP(BatchNorm1d(affine=False) → 768-128-n_cls, Dropout 0.5).
  - freeze_body=True(EMBC --freeze_body): 백본 requires_grad=False 만 끈다. model.train() 은 그대로라 bn0 배치통계·running 갱신,
    DropPath 가 살아 있다 (원본 quirk). freeze_body=False 는 툴킷 표준 태스크 전체 미세조정(논문 목표치 없음).
학습 루프(scl ce.py, split_iter 1): 배치 전체 transform → forward → CE; best-on-test Score 는 콜백이 담당.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchaudio

from .base import BaseRSCModel
from .factory import ModelFactory
from .opera import HTSATSwin, load_opera_ct_checkpoint
from rsc_toolkit.processors.opera_mel import OperaBatchMel

logger = logging.getLogger(__name__)


class TfmEncHead(nn.Module):
    """solve.py TfmEncHead — 그대로."""

    def __init__(self, embed_dim=768, output_dim=3, depth=2, heads=1, mlp_ratio=1):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.sem_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sem_blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        self.linear = nn.Linear(embed_dim, output_dim)  # apply 이후 생성 → 기본 초기화 (원본 순서)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):  # x: (B, T, D)
        x = torch.cat((self.sem_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.sem_blocks:
            x = blk(x)
        return self.linear(self.norm(x[:, 0, :]))


class MLPHead(nn.Module):
    """solve.py MLP — 그대로 (입력 [B,T,D] 를 T 평균 후 BatchNorm1d(affine=False))."""

    def __init__(self, input_size, hidden_sizes, output_size, hidden_dropout=0.5, mean=0.0, std=0.01, bias=0.0):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size, affine=False)
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        fcs = []
        for l, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
            if l > 0:
                fcs.append(nn.Dropout(hidden_dropout))
            linear = nn.Linear(i, o)
            nn.init.normal_(linear.weight, mean=mean, std=std)
            nn.init.constant_(linear.bias, bias)
            fcs.append(linear)
            fcs.append(nn.ReLU())
        self.mlp = nn.Sequential(*fcs[:-1])

    def forward(self, x):
        x = torch.mean(x, dim=-2)
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        return self.mlp(x)


@ModelFactory.register_model_class("opera_ct")
class OperaCT(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        pretrained_path: Optional[str] = None,   # encoder-operaCT.ckpt (None 이면 랜덤 초기화)
        freeze_body: bool = True,                # EMBC --freeze_body
        head: str = "tfm",                       # tfm | mlp
        head_depth: int = 4, head_heads: int = 1, head_mlp_ratio: int = 1,
        freq_mask: int = 20, time_mask: int = 40,  # scl args 기본 (evar SpecAugment, 배치 공유 마스크)
        input_sec: float = 8.0, sample_rate: int = 16000,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.htsat = HTSATSwin(drop_path_rate=drop_path_rate)
        if pretrained_path is not None:
            dropped = load_opera_ct_checkpoint(self.htsat, pretrained_path)
            logger.info("OPERA-CT ckpt 로드 %s (버린 키 %d개: torchlibrosa 프론트엔드·COLA 프로젝터)", pretrained_path, len(dropped))
        self.freeze_body = bool(freeze_body)
        if self.freeze_body:
            for p in self.htsat.parameters():
                p.requires_grad = False
        self.feature_d = self.htsat.num_features  # 768
        if head == "tfm":
            self.head = TfmEncHead(embed_dim=self.feature_d, output_dim=n_cls, depth=head_depth, heads=head_heads, mlp_ratio=head_mlp_ratio)
        elif head == "mlp":
            self.head = MLPHead(self.feature_d, (128,), n_cls)
        else:
            raise ValueError(f"head 는 tfm|mlp: {head!r}")
        self.mel = OperaBatchMel(sample_rate=sample_rate, input_sec=input_sec)
        self.freqmask = torchaudio.transforms.FrequencyMasking(freq_mask) if freq_mask > 0 else None
        self.timemask = torchaudio.transforms.TimeMasking(time_mask) if time_mask > 0 else None

    # ---- transform (배치 단위) ----
    def val_transform(self, wave: torch.Tensor) -> torch.Tensor:
        return self.mel(wave)  # (B, 64, T)

    def train_transform(self, wave: torch.Tensor) -> torch.Tensor:
        x = self.mel(wave)
        with torch.no_grad():
            if self.freqmask is not None:
                x = self.freqmask(x)
            if self.timemask is not None:
                x = self.timemask(x)
        return x

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, 64, T) → HTSAT latent (B, 768) — AR_OPERA_CT.encode_frames 의 백본 부분."""
        return self.htsat(feats.transpose(-2, -1).unsqueeze(1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(feats).unsqueeze(1))  # [B,D] → [B,1,D] 토큰 1개

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        x = self.train_transform(audio_batch) if self.training else self.val_transform(audio_batch)
        logits = self.forward(x)
        return logits, loss_func(logits, labels)
