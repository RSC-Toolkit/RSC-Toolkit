"""
PC-MCL(arXiv 2601.17080; wa976/PC-MCL) 의 AST 모델 — 공식 코드 충실 이식 (P0). Table 1 'AST + PC-MCL' (62.30).

원본 main.py set_loader/set_model/train 의 ast 분기:
  h = desired_length*100 - 2 (10 s → 998), w = 128 ; ASTModel(input_fdim=h, input_tdim=w, label_dim=3, imagenet/audioset pretrain)
  → patch-mix 계보의 **뒤집힌 fdim/tdim quirk** (실제 conv 격자는 freq 12 × time 99, pos_embed 는 (99,12) 로 보간) 를 그대로 둔다
    (rsc_toolkit/models/ast_patchmix.py PatchMixAST 를 input_fdim=998, input_tdim=128 로 재사용).
  fbank (B,1,998,128) → [학습, pcmcl 경로에서만] SpecAugment(icbhi_ast_sup, mean; 배치 단위 마스크, p≥randn quirk) → @autocast forward
  → (cls+dist)/2 768 → classifier Linear(768,3) ; ssl_classifier Linear(768,2) ; 손실 결합은 beats_pcmcl 과 동일.
  AST forward 만 @autocast (fp16) 이고 GradScaler 없음 → 모델 안에서 autocast (amp=True), trainer precision 32.
원본 ASTModel 의 미사용 파라미터(freq_attn_weights, mlp_head/domain/device 헤드) 는 forward 에 안 쓰여 학습에 영향 없다 —
PatchMixAST 가 갖는 미사용 mlp_head/classifier(LN+Linear) 도 같은 성격이라 그대로 둔다(state_dict 에는 남는다).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from rsc_toolkit.methods.pcmcl import PMHead, pcmcl_training_loss
from rsc_toolkit.methods.pcmcl_labelspace import get_label_space

from .ast_patchmix import PatchMixAST
from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)


@ModelFactory.register_model_class("ast_pcmcl")
class ASTPCMCL(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 3,
        label_mode: str = "multilabel",
        label_space: str = "icbhi4",           # icbhi4(3속성) | sprsound7(6속성)
        input_fdim: int = 998,                 # desired_length 10 s → 998 (뒤집힌 quirk 값 그대로)
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        pretrained_path: Optional[str] = None,
        input_layout: str = "freq_time",       # 툴킷 fbank (B, 128, 998)
        enable_ssl: bool = True,
        ssl_weight: float = 0.1,
        ssl_separate_training: bool = False,
        ssl_projection_dim: int = 256,
        loss_combine: str = "code",
        threshold: float = 0.5,
        specaug: bool = True,                  # 원본: --nospec 아니면 학습 시 SpecAugment (AST 실험의 실제 사용 여부 미확인 — 코드 기본 on)
        specaug_policy: str = "icbhi_ast_sup",
        specaug_mask: str = "mean",
        amp: bool = True,                      # 원본 @autocast fp16, GradScaler 없음
    ):
        super().__init__()
        if label_mode not in {"multilabel", "single"}:
            raise ValueError(f"label_mode must be multilabel|single, got {label_mode!r}")
        self.ls = get_label_space(label_space)
        if label_mode == "multilabel" and n_cls != self.ls.n_attrs:
            raise ValueError(
                f"multilabel 의 n_cls 는 라벨 공간의 속성 수와 같아야 한다 — "
                f"{self.ls.name} 는 n_cls={self.ls.n_attrs} ({', '.join(self.ls.attr_names)}), 받은 값 {n_cls}"
            )
        self.ast = PatchMixAST(
            n_cls=n_cls, fstride=10, tstride=10, input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain,
            pretrained_path=pretrained_path, method="ce", input_layout=input_layout,
        )
        self.final_feat_dim = self.ast.final_feat_dim
        self.classifier = nn.Linear(self.final_feat_dim, n_cls)     # 원본 main.py:345 — LN 없는 Linear
        self.label_mode = label_mode
        self.enable_ssl = bool(enable_ssl) and label_mode == "multilabel"
        self.pm_head = PMHead(self.final_feat_dim, separate=ssl_separate_training, projection_dim=ssl_projection_dim) if self.enable_ssl else None
        self.ssl_weight, self.loss_combine, self.threshold, self.amp = float(ssl_weight), loss_combine, float(threshold), bool(amp)
        if specaug:
            from rsc_toolkit.augmentations.specaugment import SpecAugment

            self.specaug = SpecAugment(policy=specaug_policy, mask=specaug_mask)
        else:
            self.specaug = None

    def _to_bft(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 128, 998) freq_time 또는 (B, 998, 128) time_freq → (B, F, T)."""
        if x.dim() == 4:
            x = x.squeeze(1)
        return x if self.ast.input_layout == "freq_time" else x.transpose(-2, -1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_bft = self._to_bft(x)
        if self.training and self.specaug is not None:
            x_bft = self.specaug(x_bft)                         # 원본: model(args.transforms(images)) — 배치 단위 마스크
        with torch.autocast(device_type=x_bft.device.type, dtype=torch.float16, enabled=self.amp and x_bft.is_cuda):
            feats = self.ast.forward_features(x_bft.transpose(-2, -1))   # PatchMixAST 는 (B, 998, 128) time_freq 입력
        return feats.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        features = self.forward_features(audio_batch)
        logits = self.classifier(features)
        if self.label_mode == "single":
            return logits, loss_func(logits, labels)
        if labels.dtype.is_floating_point:
            loss, _, _ = pcmcl_training_loss(logits, features, labels, loss_func, self.pm_head,
                                             meta_labels, self.ssl_weight, self.loss_combine)
            return logits, loss
        loss = loss_func(logits, self.ls.encode(labels))
        return self.ls.logits_to_onehot(logits, self.threshold), loss
