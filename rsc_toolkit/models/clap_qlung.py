"""
QLung (Audio-CLAP) — QLung 논문 Table 1·2 의 **두 번째 백본 변형**.

논문 앵커 (arXiv 2606.11915v1, Sp/Se/Score):
  ICBHI official 60/40 : 81.98±3.82 / 44.81±3.47 / **63.39±0.40**   (AST 판 62.01±1.18 보다 +1.38)
  SPRSound OOD         : 74.71±4.04 / 44.88±2.98 / **59.80±3.51**   (AST 판 58.23±3.83 보다 +1.57)

**등급 C (명세 재구현).** 저자 저장소(46408dd)에는 AST 경로만 있고 **Audio-CLAP 백본 코드가 전혀 없다**
(정찰 기록: "논문 Table 1·2 의 Audio-CLAP 백본 코드와 SPRSound OOD 평가 코드가 저장소에 전혀 없음").
따라서 이 파일은 **논문 서술 + 저자 AST 코드의 방법부**를 조합한 재구성이며, 아래 가정을 명시한다.

가정 3건 (저자 확인 대상):
  ① **방법부는 백본과 무관하게 그대로다.** AST 판의 헤드 교체(mlp_head → AngularClassifier s_a=37)와
     DFAM(s_d=15, κ=0.5, γ=0.5, m_target=0.2, λ=0.4)를 CLAP 특징에 그대로 얹는다.
     근거: 논문이 두 백본을 같은 방법의 변형으로 제시하고 하이퍼파라미터를 따로 밝히지 않는다.
  ② **특징은 CLAP audio_embeds(512-d)** 를 쓴다 — Audio-CLAP 기준선(models/clap.py)과 동일 지점.
     `qlung_dual_loss` 가 F.normalize 를 걸므로 정규화 여부는 방법부가 정한다.
  ③ **AQS 는 백본에 들어가는 파형에서 계산**한다. CLAP 경로는 48 kHz 이므로 AST 판(16 kHz)과
     AQS 값이 달라진다 — 논문은 AQS 계산의 샘플링레이트를 밝히지 않는다.

`method=ce` 로 두면 AngularClassifier 없이 Audio-CLAP 기준선과 같은 선형 헤드가 되어,
같은 데이터 경로에서 (방법 − 기준선) 이득을 잴 수 있다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .backbone.clap import PretrainedCLAPWithProjection
from .factory import ModelFactory
from rsc_toolkit.methods.qlung import (
    ICBHI_TRAIN_CLASS_NUMS,
    AngularClassifier,
    AngularMarginLoss,
    qlung_dual_loss,
)


@ModelFactory.register_model_class("clap_qlung")
class QLungCLAP(BaseRSCModel, nn.Module):
    # MA(live weight averaging) 제외 접두어 — 원본이 DFAM 을 MA 하지 않는 것과 같다.
    ma_exclude_prefixes = ("dfam.",)
    accepts_meta = ("qlung_aqs",)  # batch[2] = AQS float (QLungDataModule)

    def __init__(
        self,
        n_cls: int = 4,
        method: str = "aqm",                 # "aqm" = QLung | "ce" = Audio-CLAP 기준선(같은 데이터 경로)
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feat_dim: int = 512,
        s_a: float = 37.0,                   # AngularClassifier scale (train_qlung.py:271)
        s_d: float = 15.0,                   # DFAM scale (train_qlung.py:328)
        kappa: float = 0.5,                  # DFAM m (품질 마진 계수)
        gamma: float = 0.5,
        m_target: float = 0.2,
        lambda_aqm: float = 0.4,             # --alpha (DFAM CE 가중)
        class_nums=None,                     # π_y — 데이터셋 train 분포. 미지정이면 ICBHI 기본
        easy_margin: bool = False,
    ):
        super().__init__()
        if method not in {"aqm", "ce"}:
            raise ValueError(f"method must be aqm|ce, got {method!r}")
        self.method = method
        self.lambda_aqm = float(lambda_aqm)
        self.final_feat_dim = int(final_feat_dim)

        self.backbone = PretrainedCLAPWithProjection(
            pretrained_name=pretrained_name,
            final_feat_dim=final_feat_dim,
        )
        if method == "aqm":
            self.classifier = AngularClassifier(self.final_feat_dim, n_cls, s=s_a)
            self.dfam = AngularMarginLoss(
                self.final_feat_dim, n_cls, s=s_d, m=kappa, gamma=gamma, m_target=m_target,
                class_nums=list(class_nums) if class_nums is not None else list(ICBHI_TRAIN_CLASS_NUMS),
                easy_margin=easy_margin,
            )
        else:
            self.classifier = nn.Linear(self.final_feat_dim, n_cls)
            self.dfam = None

    # ------------------------------------------------------------------ 특징
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """CLAP audio_embeds (B, 512). L2 정규화는 하지 않는다 — 방법부가 건다."""
        return self.backbone(x)

    def eval_logits(self, x: torch.Tensor) -> torch.Tensor:
        """평가 경로: 정규화 특징 → AngularClassifier (원본 validate:541-543 와 같은 규약)."""
        features = self.forward_features(x)
        if self.method == "ce":
            return self.classifier(features)
        return self.classifier(torch.nn.functional.normalize(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_logits(x)

    # ------------------------------------------------------------------ 학습
    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None, **kwargs):
        """
        meta_labels = AQS float (B,) — aqm 학습에 필수. loss_func = 비가중 CE.
        학습·검증이 같은 메서드를 부르므로 self.training 으로 분기한다(base.py 계약).
        """
        if not self.training:
            logits = self.eval_logits(audio_batch)
            cast = logits.to(torch.float64) if self.method == "aqm" else logits
            return logits, loss_func(cast, labels)

        features = self.forward_features(audio_batch)

        if self.method == "ce":
            logits = self.classifier(features)
            return logits, loss_func(logits, labels)

        if meta_labels is None:
            raise ValueError(
                "clap_qlung(method=aqm) 학습에는 meta_labels 자리에 AQS float 텐서가 필요하다 "
                "(data=icbhi_qlung_clap)"
            )
        quality = meta_labels.to(features.device)
        if quality.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"AQS 는 float 텐서여야 한다: {quality.dtype}")
        logits, loss, _, _ = qlung_dual_loss(
            features, labels, quality, self.classifier, self.dfam, loss_func,
            lambda_aqm=self.lambda_aqm,
        )
        return logits, loss
