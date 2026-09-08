"""
PAFA 논문(Interspeech 2025)의 BEATs 미세조정 모델 — 공식 코드 충실 이식 (P0).

출처:
- 논문: Patient-Aware Feature Alignment for Robust Lung Sound Classification:
  Cohesion-Separation and Global Alignment Losses (https://arxiv.org/abs/2505.23834)
- 공식 코드: https://github.com/wa976/PAFA models/beats.py (BEATsTransferLearningModel)
  + main.py set_model()/train()/validate() 의 beats 분기.

구조 (원본 그대로):
- 백본: BEATs(iter3+ AS2M 사전학습 ckpt 의 cfg 로 구성, predictor 없음) — 입력은
  16 kHz 파형 (B, L). extract_features 가 내부에서 kaldi fbank(128 mel, 25/10 ms) →
  패치 임베딩 → Transformer 로 토큰 시퀀스 (B, T', 768) 를 낸다.
- 분류기: nn.Linear(768, n_cls) 를 **토큰별로** 적용한 뒤 시간축 평균 (main.py:365-367).
- PAFA: ProjectionHead(768→768, attention 풀링, LayerNorm) 로 (B, 768) 을 만들고
  PAFALoss(patient_ids) 를 더한다: loss = w_ce*CE + w_pafa*PAFA (main.py:379-389).
- SpecAugment 는 스크립트가 --nospec 이라 없음 (spec_transform=None).

툴킷 배선: datamodule 이 (파형, 라벨, 환자 ID) 배치를 준다 (meta_label_mode=patient).
"""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .beats_upstream import BEATs, BEATsConfig
from .factory import ModelFactory

logger = logging.getLogger(__name__)

BEATS_CKPT_CANDIDATES = (
    "pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt",
    "pretrained_models/BEATs_iter3_plus_AS2M.pt",  # 원본 코드가 쓰는 위치
)


def _resolve_beats_ckpt(path: Optional[str]) -> pathlib.Path:
    from rsc_toolkit.paths import resolve_pretrained

    if path is not None:
        return resolve_pretrained(explicit=path)
    found = resolve_pretrained(BEATS_CKPT_CANDIDATES)  # RSCTK_PRETRAINED_ROOT → cwd 순
    if found is not None:
        return found
    raise FileNotFoundError(
        "BEATs_iter3_plus_AS2M.pt 가 없다. microsoft/unilm/beats 의 'BEATs_iter3+ (AS2M)' 를 받아 "
        f"{BEATS_CKPT_CANDIDATES[0]} 에 둘 것"
    )


class BEATsTransferLearningModel(nn.Module):
    """원본 models/beats.py 와 동일 (ckpt cfg 로 BEATs 구성 + 가중치 로드, extract_features 반환)."""

    def __init__(self, num_target_classes: int = 4, model_path: Optional[str] = None, spec_transform=None):
        super().__init__()
        self.num_target_classes = num_target_classes
        ckpt_path = _resolve_beats_ckpt(model_path)
        self.checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = BEATsConfig({
            **self.checkpoint["cfg"],
            "predictor_class": self.num_target_classes,
            "finetuned_model": False,
            "spec_transform": spec_transform,
        })
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])
        self.final_feat_dim = self.cfg.encoder_embed_dim
        del self.checkpoint  # 원본은 들고 있지만 state_dict 에 안 들어가는 잔여 메모리라 버린다

    def forward(self, x, padding_mask=None, training=False):
        if padding_mask is not None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)
        return x


@ModelFactory.register_model_class("beats")
class BEATsClassifier(BaseRSCModel, nn.Module):
    """원본 set_model 의 beats 구성: 백본 + 토큰별 Linear(mean 풀링) [+ PAFA ProjectionHead]."""

    accepts_meta = ("patient",)  # batch[2] = 환자 번호 (meta_label_mode=patient; method=pafa 에 필수)

    def __init__(
        self,
        n_cls: int = 4,
        pretrained_path: Optional[str] = None,
        method: str = "ce",             # "ce" | "pafa"
        # ---- PAFA (스크립트 beats_pafa.sh 값이 기본) ----
        w_ce: float = 1.0,
        w_pafa: float = 1.0,
        lambda_pcsl: float = 50.0,
        lambda_gpal: float = 0.0005,
        hidden_dim: Optional[int] = None,
        output_dim: int = 768,
        norm_type: str = "ln",
        proj_type: str = "end2end",
    ):
        super().__init__()
        if method not in {"ce", "pafa"}:
            raise ValueError(f"method must be ce|pafa, got {method!r}")
        self.backbone = BEATsTransferLearningModel(num_target_classes=n_cls, model_path=pretrained_path)
        self.classifier = nn.Linear(self.backbone.final_feat_dim, n_cls)
        self.method = method
        self.w_ce, self.w_pafa = w_ce, w_pafa
        self.lambda_pcsl, self.lambda_gpal = lambda_pcsl, lambda_gpal
        if method == "pafa":
            from rsc_toolkit.methods.pafa import PAFALoss, ProjectionHead

            # 원본 main.py:250 — beats 는 attention=True
            self.projector = ProjectionHead(
                self.backbone.final_feat_dim, hidden_dim, output_dim,
                attention=True, norm_type=norm_type, proj_type=proj_type,
            )
            self.pafa_loss = PAFALoss()
        else:
            self.projector = nn.Identity()
            self.pafa_loss = None

    def forward_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (B, 1, L) -> (B, L)
        return self.backbone(waveform)      # (B, T', 768)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 원본: output = classifier(features); output = output.mean(dim=1)
        return self.classifier(self.forward_features(waveform)).mean(dim=1)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        features = self.forward_features(audio_batch)
        logits = self.classifier(features).mean(dim=1)
        loss_class = loss_func(logits, labels)
        if (not self.training) or self.method == "ce":
            return logits, loss_class
        if meta_labels is None:
            raise ValueError("method=pafa 는 배치에 환자 ID 가 필요하다 (datamodule meta_label_mode=patient)")
        proj = self.projector(features)
        loss_pafa = self.pafa_loss(proj, meta_labels, lambda_pcsl=self.lambda_pcsl, lambda_gpal=self.lambda_gpal)
        return logits, self.w_ce * loss_class + self.w_pafa * loss_pafa
