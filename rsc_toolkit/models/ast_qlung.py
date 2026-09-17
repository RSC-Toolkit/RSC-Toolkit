# Portions of this file are ported from QLung (https://github.com/RSC-Toolkit/QLung), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
AST of QLung (Interspeech 2026) — faithful port of the official code (RSC-Toolkit/QLung 46408dd) (exact port of the official code, verified against goldens).

Sources: train_qlung.py set_model()/train()/validate(), method/loss.py, models/ast.py (= byte-identical to models/ast.py of the sgscl clone;
the non-patch-mix forward path is functionally identical to PatchMixAST — pinned by the golden tests/golden/qlung/model_golden.pt).

Architecture (method="aqm", default):
  AST (IN+AS, swapped 798×128 quirk) → (cls+dist)/2 features → F.normalize → AngularClassifier(s_a=37) logits (CE in float64)
                                                   └→ DFAM AngularMarginLoss(s_d=15, κ=0.5, γ=0.5, m_target=0.2, π_y=ICBHI train counts)
  Total loss = CE(logits.float64) + λ(0.4)·CE(DFAM logits).  Evaluation also uses normalized features → AngularClassifier (validate :541-543).
  The DFAM weight W_d is registered as the submodule `dfam` → included in the optimizer (train_qlung.py:345). It must be excluded from MA(0.6)
  (train_qlung.py:490-499 covers only model, classifier, projector) → callbacks/qlung_ma_update.py.
  The unused mlp_head remains as in the original (present in state_dict and optimizer, receives no gradient).

method="ce" (CE-path quirk of the code kept as-is; differs from the original Patch-Mix code):
  train: classifier(features)  — no normalization (train_qlung.py:459)
  eval : classifier(F.normalize(features)) — normalized (train_qlung.py:541-542), loss uses output.to(float64)
  classifier = deepcopy(mlp_head) (LayerNorm+Linear).

SpecAugment: the original train() applies args.transforms(images) once to the whole batch (icbhi_ast_sup, p ≥ torch.randn(1) gate) — not in evaluation.
AQS (float, (B,)) arrives in the meta_labels slot of training_forward (datasets/qlung_data.py).
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
    """Original ASTModel + set_model (method aqm|ce) setup."""

    accepts_meta = ("qlung_aqs",)  # batch[2] = AQS float (QLungDataModule)

    # state_dict prefixes excluded from MA (live weight averaging) — the original does not apply MA to criterion[1] (DFAM).
    ma_exclude_prefixes = ("dfam.",)

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 798,           # intentionally swapped quirk (see ast_patchmix.py docstring)
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        # ---- QLung arguments (train_qlung.py / scripts/train_qlung_icbhi.sh) ----
        method: str = "aqm",             # "aqm" | "ce"
        s_a: float = 37.0,               # AngularClassifier scale (train_qlung.py:271)
        s_d: float = 15.0,               # DFAM scale (train_qlung.py:328)
        kappa: float = 0.5,              # DFAM m (quality margin coefficient, train_qlung.py:328 m=0.5)
        gamma: float = 0.5,              # --gamma
        m_target: float = 0.2,           # --m_target
        lambda_aqm: float = 0.4,         # --alpha (DFAM CE weight)
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
            # set_model: classifier = AngularClassifier(final_feat_dim, n_cls, s=37) — the mlp_head copy is discarded
            self.classifier = AngularClassifier(self.final_feat_dim, n_cls, s=s_a)
            self.dfam = AngularMarginLoss(self.final_feat_dim, n_cls, s=s_d, m=kappa, gamma=gamma, m_target=m_target,
                                          class_nums=list(class_nums), easy_margin=easy_margin)
        else:
            # keep the parent's classifier = deepcopy(mlp_head)
            self.dfam = None
        # original train(): applies args.transforms = SpecAugment(args) to the batch (not in evaluation)
        self.specaug = SpecAugment(policy=specaug_policy, mask=specaug_mask)

    # ------------------------------------------------------------------ paths
    def eval_logits(self, x_paper: torch.Tensor) -> torch.Tensor:
        """validate(): features → F.normalize → classifier (shared by aqm and ce, train_qlung.py:540-542)."""
        return self.classifier(F.normalize(self.forward_features(x_paper)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_logits(self._to_paper_layout(x))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """
        meta_labels = AQS float (B,) — required for aqm training. loss_func = unweighted CE (criterion[0]).
        Returns (logits, loss). Eval mode follows the original validate path (normalized features, loss = CE(logits.float64)).
        """
        x = self._to_paper_layout(audio_batch)  # (B, 798, 128) = (B, time, freq)

        if not self.training:
            logits = self.eval_logits(x)
            return logits, loss_func(logits.to(torch.float64), labels)

        x = apply_specaug_paper(self.specaug, x)
        features = self.forward_features(x)

        if self.method == "ce":
            logits = self.classifier(features)          # quirk: training uses unnormalized features (train_qlung.py:459)
            return logits, loss_func(logits, labels)

        if meta_labels is None:
            raise ValueError("ast_qlung(method=aqm) training requires an AQS float tensor in the meta_labels slot (data=icbhi_qlung)")
        quality = meta_labels.to(features.device)
        if quality.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"AQS must be a float tensor: {quality.dtype}")

        logits, loss, _, _ = qlung_dual_loss(features, labels, quality, self.classifier, self.dfam, loss_func,
                                             lambda_aqm=self.lambda_aqm)
        return logits, loss
