"""
PC-MCL(arXiv 2601.17080; wa976/PC-MCL) 의 BEATs 모델 — 공식 코드 충실 이식 (P0). 주 결과(65.37) 백본.

원본 main.py set_model/train/validate 의 beats 분기:
  features = BEATsTransferLearningModel(...)(images)          # extract_features → (B, T', 768)   (models/BEATs/beats.py)
  features = features.mean(dim=1)                             # 토큰 평균 (main.py:526)  ※ PAFA 는 토큰별 Linear 후 평균 — 다르다
  output   = classifier(features)                             # nn.Linear(768, n_cls=3) (main.py:345)
  ssl_out  = ssl_classifier(features[ssl_label >= 0])          # nn.Linear(768, 2) (:368; --ssl_separate_training 이면 projector 앞에)
  loss     = (1 - ssl_weight) * BCEWithLogits(output, labels) + ssl_weight * CE(ssl_out, ssl_label)   (:563-586)
SpecAugment: --nospec 이 아니면 SpecAugment(args) 를 BEATs 내부 fbank 에 적용 (:319-322). 스크립트 부재 → 기본 off (가정).
AMP 없음(main.py 에 autocast 없음) → trainer precision 32.
평가(validate): 테스트 사이클 단독 → sigmoid > threshold → conversion.py 규칙으로 4-class → Sp/Se/Score.

툴킷 배선: 학습 배치 = (파형 (B,L), 3-label float (B,3), ssl_label long (B)) — PLAdapter 가 batch[2] 를 meta_labels 로 넘긴다.
검증 배치 = (파형, 4-class long) — 4→3 변환 후 BCE 를 val_loss 로, 반환 logits 는 3-sigmoid→4-class one-hot (B,4)
(ICBHIScore 의 argmax 와 맞도록; 메트릭 레지스트리 수정 없음).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from rsc_toolkit.methods.pcmcl import (
    PMHead,
    label4_to_multilabel,
    multilabel_logits_to_onehot4,
    pcmcl_training_loss,
)

from .base import BaseRSCModel
from .beats_pafa import BEATsTransferLearningModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)


class _BEATsFbankSpecAugment(nn.Module):
    """원본 SpecAugment(args) 를 BEATs fbank [B,1,T,F] 에 적용: 원본은 squeeze→transpose 로 [B,F,T] 에 마스크한다."""

    def __init__(self, policy: str = "icbhi_ast_sup", mask: str = "mean"):
        super().__init__()
        from rsc_toolkit.augmentations.specaugment import SpecAugment

        self.aug = SpecAugment(policy=policy, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, T, F)
        y = self.aug(x.squeeze(1).transpose(1, 2))       # (B, F, T)
        return y.transpose(1, 2).unsqueeze(1)


@ModelFactory.register_model_class("beats_pcmcl")
class BEATsPCMCL(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 3,                        # 3-label (default/noconcat_multi) | 4 (ablation concat_single, CE)
        label_mode: str = "multilabel",        # multilabel | single
        pretrained_path: Optional[str] = None,
        enable_ssl: bool = True,               # PM 헤드 (Table 3 'Concat+Multi' ablation 은 false)
        ssl_weight: float = 0.1,               # 논문 α=0.1 ↔ 코드 --ssl_weight (argparse 기본 0.5) — 가정
        ssl_separate_training: bool = False,
        ssl_projection_dim: int = 256,
        loss_combine: str = "code",            # code: (1-w)·main + w·ssl | paper: main + w·ssl
        threshold: float = 0.5,
        specaug: bool = False,                 # 원본 기본은 on(--nospec 아니면) — 스크립트 부재로 PAFA 계보 nospec 가정
        specaug_policy: str = "icbhi_ast_sup",
        specaug_mask: str = "mean",
    ):
        super().__init__()
        if label_mode not in {"multilabel", "single"}:
            raise ValueError(f"label_mode must be multilabel|single, got {label_mode!r}")
        if label_mode == "multilabel" and n_cls != 3:
            raise ValueError("multilabel 은 n_cls=3 (normal, crackle, wheeze)")
        spec_transform = _BEATsFbankSpecAugment(specaug_policy, specaug_mask) if specaug else None
        self.backbone = BEATsTransferLearningModel(num_target_classes=n_cls, model_path=pretrained_path, spec_transform=spec_transform)
        self.final_feat_dim = self.backbone.final_feat_dim
        self.classifier = nn.Linear(self.final_feat_dim, n_cls)
        self.label_mode = label_mode
        self.enable_ssl = bool(enable_ssl) and label_mode == "multilabel"
        self.pm_head = PMHead(self.final_feat_dim, separate=ssl_separate_training, projection_dim=ssl_projection_dim) if self.enable_ssl else None
        self.ssl_weight, self.loss_combine, self.threshold = float(ssl_weight), loss_combine, float(threshold)

    def forward_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        feats = self.backbone(waveform)                     # (B, T', 768)
        if feats.dim() == 3:
            feats = feats.mean(dim=1)                      # (B, 768)
        return feats

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        features = self.forward_features(audio_batch)
        logits = self.classifier(features)
        if self.label_mode == "single":
            return logits, loss_func(logits, labels)
        if labels.dtype.is_floating_point:                 # 학습 배치: 3-label (+ PM 라벨)
            loss, _, _ = pcmcl_training_loss(logits, features, labels, loss_func, self.pm_head,
                                             meta_labels, self.ssl_weight, self.loss_combine)
            return logits, loss
        # 검증 배치: 4-class long → 3-label BCE(원본 validate 의 criterion(output, labels)) + 4-class one-hot 예측
        loss = loss_func(logits, label4_to_multilabel(labels))
        return multilabel_logits_to_onehot4(logits, self.threshold), loss
