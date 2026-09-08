"""
SCL-metadata(Moummad & Farrugia, ICASSP 2023; arXiv 2210.16192) — 공식 코드 충실 이식 (P0).

출처: https://github.com/ilyassmoummad/scl_icbhi2017 (models.py, losses.py, ce.py, scl.py, mscl.py, hybrid.py, main.py)

구성 (원본 그대로):
- 인코더 CNN6 (PANN ConvBlock5x5 x4, AudioSet Cnn6_mAP=0.343 부분 로드, dropout 없음) → 512-d.
  툴킷 sft_backbones.CNN6SFT 와 블록·풀링·부분 로드가 동일하다 (입력 [B,1,F,T] 로 넣으면 시간 평균 → 주파수 max+mean).
- 배치 변환(main.py:59-67, 모델 안에서 수행): dB-mel(프로세서가 캐시) → 배치 전체 min-max → [SpecAugment(train)]
  → Standardize(0.3690, 0.0255).
- method
  ce     : Linear(512, n_cls), 단일 뷰(SpecAug 1회), 가중 CE (loss_func) — ce.py:22-28
  hybrid : 두 뷰 → α·SupCon(proj1, proj2, y) + (1-α)·CE(cat[pred1,pred2], cat[y,y]) — losses.py:46-57, hybrid.py
  scl    : 두 뷰 → SupCon(proj1, proj2, y), τ 0.06 — scl.py (사전학습, 평가 없음, 마지막 epoch 상태 저장)
  mscl   : 두 뷰, projector 2개 → (1-λ)·SupCon(p1,p2, meta) + λ·SupCon(p1',p2', y), λ 0.75 — mscl.py:15-19
  linear : encoder_ckpt 로드 후 동결(requires_grad False + eval; BN 통계 고정), val_transform(SpecAug 없음)
           특징으로 Linear(512, n_cls) 만 학습, 가중 CE — scl.py:150-152, linear 단계
- Projector: Linear(512,512)-ReLU-Linear(512,128), BN 없음 (models.py:256-271, apply_bn=False).
- SupCon 손실은 methods/sgscl.py MetaCL 과 수식이 같아 재사용한다 (labels=None 이면 SimCLR).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .projector import Projector
from .sft_backbones import CNN6SFT
from rsc_toolkit.augmentations.specaugment_torchaudio import SpecAugmentTA
from rsc_toolkit.methods.sgscl import MetaCL as SupConLoss
from rsc_toolkit.processors.sclmeta_mel import STANDARDIZE_MEAN, STANDARDIZE_STD, batch_minmax, standardize

logger = logging.getLogger(__name__)

METHODS = ("ce", "hybrid", "scl", "mscl", "linear")


@ModelFactory.register_model_class("cnn6_sclmeta")
class CNN6SCLMeta(BaseRSCModel, nn.Module):
    accepts_meta = ("sa", "sa_sprs")  # batch[2] = 성별×연령군 클래스 (SCL-metadata; method=mscl 에 필수)

    def __init__(
        self,
        n_cls: int = 4,
        method: str = "mscl",
        from_sl_official: bool = True,     # PANN AudioSet 부분 로드 (--scratch 면 False)
        out_dim: int = 128,
        temperature: float = 0.06,
        alpha: float = 0.5,                # hybrid: SupCon 비중
        lam: float = 0.75,                 # mscl: class-SCL 비중 (meta 는 1-λ)
        freq_mask: int = 20, time_mask: int = 40, freq_stripes: int = 2, time_stripes: int = 2, specaug_p: float = 1.0,
        std_mean: float = STANDARDIZE_MEAN, std_std: float = STANDARDIZE_STD,
        encoder_ckpt: Optional[str] = None,  # linear: 사전학습 run 의 final_state ("model" 키 — final_state.py) 경로
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.alpha, self.lam = alpha, lam
        self.std_mean, self.std_std = std_mean, std_std
        self.encoder = CNN6SFT()
        if from_sl_official and method != "linear":
            self.encoder.load_sl_official_weights()
        self.final_feat_dim = self.encoder.final_feat_dim
        self.specaug = SpecAugmentTA(freq_mask, time_mask, freq_stripes, time_stripes, specaug_p)
        self.supcon = SupConLoss(temperature=temperature)

        self.classifier = nn.Linear(self.final_feat_dim, n_cls) if method in ("ce", "hybrid", "linear") else None
        self.projector = Projector(self.final_feat_dim, out_dim, apply_bn=False) if method in ("hybrid", "scl", "mscl") else None
        self.projector2 = Projector(self.final_feat_dim, out_dim, apply_bn=False) if method == "mscl" else None

        if method == "linear":
            if encoder_ckpt is None:
                raise ValueError("method=linear 에는 encoder_ckpt (사전학습 run 의 final_state.pth) 가 필요하다")
            state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            enc = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc, strict=True)  # strict 라 불일치는 예외로 드러난다
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("선형 평가용 인코더 로드·동결: %s", encoder_ckpt)

    # 원본 linear 단계: encoder.eval() 고정 (BN 러닝 통계 갱신 안 함)
    def train(self, mode: bool = True):
        super().train(mode)
        if self.method == "linear":
            self.encoder.eval()
        return self

    # ----------------------------------------------------------- 배치 변환
    def _as_4d(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 3 else x  # (B, F, T) -> (B, 1, F, T)

    def _val_transform(self, x):
        return standardize(batch_minmax(self._as_4d(x)), self.std_mean, self.std_std)

    def _train_transform(self, x):
        with torch.no_grad():
            return standardize(self.specaug(batch_minmax(self._as_4d(x))), self.std_mean, self.std_std)

    def forward_features(self, x_transformed: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_transformed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError(f"method={self.method} 에는 분류기가 없다 (사전학습 전용)")
        return self.classifier(self.forward_features(self._val_transform(x)))

    # ---------------------------------------------------------------- 학습
    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if not self.training or self.method == "linear":
            if self.classifier is None:
                # scl/mscl 사전학습 run 은 평가가 없다 (config 에서 limit_val_batches=0); 방어적 더미
                z = torch.zeros(audio_batch.shape[0], 4, device=audio_batch.device)
                return z, torch.zeros((), device=audio_batch.device, requires_grad=self.training)
            logits = self.forward(audio_batch)
            return logits, loss_func(logits, labels)

        if self.method == "ce":
            logits = self.classifier(self.forward_features(self._train_transform(audio_batch)))
            return logits, loss_func(logits, labels)

        v1, v2 = self._train_transform(audio_batch), self._train_transform(audio_batch)
        f1, f2 = self.forward_features(v1), self.forward_features(v2)

        if self.method == "hybrid":
            p1, p2 = self.projector(f1), self.projector(f2)
            pred1, pred2 = self.classifier(f1), self.classifier(f2)
            loss = self.alpha * self.supcon(p1, p2, labels) + (1 - self.alpha) * loss_func(
                torch.cat([pred1, pred2], dim=0), torch.cat([labels, labels], dim=0))
            return pred1, loss
        if self.method == "scl":
            return torch.zeros(f1.shape[0], 4, device=f1.device), self.supcon(self.projector(f1), self.projector(f2), labels)
        # mscl
        if meta_labels is None:
            raise ValueError("method=mscl 에는 meta_labels(sa) 가 필요하다 — datamodule meta_label_mode=sa")
        p1, p2 = self.projector(f1), self.projector(f2)
        q1, q2 = self.projector2(f1), self.projector2(f2)
        loss = (1.0 - self.lam) * self.supcon(p1, p2, meta_labels) + self.lam * self.supcon(q1, q2, labels)
        return torch.zeros(f1.shape[0], 4, device=f1.device), loss
