"""
QLung(Interspeech 2026)의 AST — 공식 코드(RSC-Toolkit/QLung 46408dd) 충실 이식 (P0).

출처: train_qlung.py set_model()/train()/validate(), method/loss.py, models/ast.py (= sgscl 클론 models/ast.py 와 바이트 동일;
비-patch-mix forward 경로는 PatchMixAST 와 기능 동일 — 골든 tests/golden/qlung/model_golden.pt 로 고정).

구조 (method="aqm", 기본):
  AST(IN+AS, 뒤집힌 798×128 quirk) → (cls+dist)/2 특징 → F.normalize → AngularClassifier(s_a=37) 로짓 (float64 로 CE)
                                                   └→ DFAM AngularMarginLoss(s_d=15, κ=0.5, γ=0.5, m_target=0.2, π_y=ICBHI train 개수)
  총손실 = CE(로짓.float64) + λ(0.4)·CE(DFAM 로짓).  평가도 정규화 특징 → AngularClassifier (validate :541-543).
  DFAM 가중치 W_d 는 서브모듈 `dfam` 으로 등록 → optimizer 에 포함(train_qlung.py:345). MA(0.6) 에서는 제외해야 한다
  (train_qlung.py:490-499 는 model·classifier·projector 만) → callbacks/qlung_ma_update.py.
  미사용 mlp_head 는 원본처럼 잔존(state_dict·optimizer 에 있고 gradient 없음).

method="ce" (코드의 CE 경로 quirk 그대로, Patch-Mix 원 코드와 다름):
  train: classifier(features)  — 정규화 없음 (train_qlung.py:459)
  eval : classifier(F.normalize(features)) — 정규화 (train_qlung.py:541-542), loss 는 output.to(float64)
  classifier = deepcopy(mlp_head) (LayerNorm+Linear).

SpecAugment: 원본 train() 은 args.transforms(images) 를 배치 전체에 1회 적용(icbhi_ast_sup, p ≥ torch.randn(1) 게이트) — 평가 없음.
training_forward 의 meta_labels 자리로 AQS(float, (B,)) 가 온다 (datasets/qlung_data.py).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ast_patchmix import PatchMixAST
from .factory import ModelFactory
from rsc_toolkit.augmentations.specaugment import SpecAugment
from rsc_toolkit.methods.qlung import (
    ICBHI_TRAIN_CLASS_NUMS,
    AngularClassifier,
    AngularMarginLoss,
    qlung_dual_loss,
)
from rsc_toolkit.methods.sgscl import apply_specaug_paper


@ModelFactory.register_model_class("ast_qlung")
class QLungAST(PatchMixAST):
    """원본 ASTModel + set_model(method aqm|ce) 구성."""

    accepts_meta = ("qlung_aqs",)  # batch[2] = AQS float (QLungDataModule)

    # MA(live weight averaging) 에서 제외할 state_dict 접두어 — 원본은 criterion[1](DFAM) 을 MA 하지 않는다.
    ma_exclude_prefixes = ("dfam.",)

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 798,           # 일부러 뒤집힌 quirk (ast_patchmix.py docstring)
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        # ---- QLung 인자 (train_qlung.py / scripts/train_qlung_icbhi.sh) ----
        method: str = "aqm",             # "aqm" | "ce"
        s_a: float = 37.0,               # AngularClassifier scale (train_qlung.py:271)
        s_d: float = 15.0,               # DFAM scale (train_qlung.py:328)
        kappa: float = 0.5,              # DFAM m (품질 마진 계수, train_qlung.py:328 m=0.5)
        gamma: float = 0.5,              # --gamma
        m_target: float = 0.2,           # --m_target
        lambda_aqm: float = 0.4,         # --alpha (DFAM CE 가중)
        class_nums: Sequence[float] = ICBHI_TRAIN_CLASS_NUMS,
        easy_margin: bool = False,
        specaug_policy: str = "icbhi_ast_sup",
        specaug_mask: str = "mean",
        input_layout: str = "time_freq",
    ):
        super().__init__(
            n_cls=n_cls, fstride=fstride, tstride=tstride,
            input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain,
            model_size=model_size, pretrained_path=pretrained_path, verbose=verbose,
            method="ce", input_layout=input_layout,
        )
        if method not in {"aqm", "ce"}:
            raise ValueError(f"method must be aqm|ce, got {method!r}")
        self.method = method
        self.lambda_aqm = float(lambda_aqm)
        self.n_cls = n_cls
        if method == "aqm":
            # set_model: classifier = AngularClassifier(final_feat_dim, n_cls, s=37) — mlp_head 복사본은 버려진다
            self.classifier = AngularClassifier(self.final_feat_dim, n_cls, s=s_a)
            self.dfam = AngularMarginLoss(self.final_feat_dim, n_cls, s=s_d, m=kappa, gamma=gamma, m_target=m_target,
                                          class_nums=list(class_nums), easy_margin=easy_margin)
        else:
            # 부모가 만든 classifier = deepcopy(mlp_head) 그대로
            self.dfam = None
        # 원본 train(): args.transforms = SpecAugment(args) 를 배치에 적용 (평가 제외)
        self.specaug = SpecAugment(policy=specaug_policy, mask=specaug_mask)

    # ------------------------------------------------------------------ 경로
    def eval_logits(self, x_paper: torch.Tensor) -> torch.Tensor:
        """validate(): features → F.normalize → classifier (aqm·ce 공통, train_qlung.py:540-542)."""
        return self.classifier(F.normalize(self.forward_features(x_paper)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_logits(self._to_paper_layout(x))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """
        meta_labels = AQS float (B,) — aqm 학습에 필수. loss_func = 비가중 CE(criterion[0]).
        반환 (logits, loss). 평가 모드는 원본 validate 경로(정규화 특징, loss = CE(logits.float64)).
        """
        x = self._to_paper_layout(audio_batch)  # (B, 798, 128) = (B, time, freq)

        if not self.training:
            logits = self.eval_logits(x)
            return logits, loss_func(logits.to(torch.float64), labels)

        x = apply_specaug_paper(self.specaug, x)
        features = self.forward_features(x)

        if self.method == "ce":
            logits = self.classifier(features)          # quirk: 학습은 비정규화 특징 (train_qlung.py:459)
            return logits, loss_func(logits, labels)

        if meta_labels is None:
            raise ValueError("ast_qlung(method=aqm) 학습에는 meta_labels 자리에 AQS float 텐서가 필요하다 (data=icbhi_qlung)")
        quality = meta_labels.to(features.device)
        if quality.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"AQS 는 float 텐서여야 한다: {quality.dtype}")
        logits, loss, _, _ = qlung_dual_loss(features, labels, quality, self.classifier, self.dfam, loss_func,
                                             lambda_aqm=self.lambda_aqm)
        return logits, loss
