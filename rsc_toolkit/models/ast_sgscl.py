# Portions of this file are ported from SG-SCL (https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
AST of the SG-SCL paper (ICASSP 2024) — faithful port of the official code (exact port of the official code, verified against goldens).

Sources:
- Official code: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
  models/ast.py + main.py set_model()/train().
- The AST skeleton, pos_embed adaptation, and forward are literally identical to patch-mix's ast.py, so
  PatchMixAST is inherited and reused (identity pinned by goldens). Only the sgscl deltas are added:
  domain_mlp_head, domain classifier/Projector, GRL, MetaCL, batch-level SpecAugment applied twice.

Quirks kept from the original:
- domain_mlp_head dimension: set_model passes domain_label_dim=m_cls only with
  --domain_adaptation (DAT). For SG-SCL (domain_adaptation2) and CE it is built with the default
  527 (AudioSet) and stays in the state_dict **without ever being used**.
- SpecAugment is applied to the whole batch in the train loop (one mask set per batch). Because the
  implementation is in-place, the second SG-SCL view accumulates masks on top of the first
  (see the methods/sgscl.py docstring). Not applied in evaluation.
- Use of GRL (features[1]) in SG-SCL differs by repository version: the clone 6656460 (2024-12-05)
  used as the golden reference uses features[0] (no GRL) for the contrastive features, and the later GitHub
  HEAD commit 2ed1bac switched to features[1] (through GRL). The default con_feat="feat0" is the
  clone behavior, con_feat="feat1" is GitHub HEAD, and cls_feat="feat1" is an experimental flag that
  also routes the classifier input through GRL. GRL has no parameters, so the state_dict and
  evaluation path are identical for every combination (existing ckpts stay compatible).
- The original's unused projector (only constructed, then mixed with snapshots via MA) is not reproduced.
"""
from __future__ import annotations

from copy import deepcopy

import torch.nn as nn

from .ast_patchmix import PatchMixAST
from .factory import ModelFactory
from .projector import Projector
from rsc_toolkit.augmentations.specaugment import SpecAugment
from rsc_toolkit.methods.sgscl import MetaCL, sgscl_training_losses


@ModelFactory.register_model_class("ast_sgscl")
class SGSCLAST(PatchMixAST):
    """Original sgscl ASTModel + set_model setup (method: ce | dat | sgscl)."""

    accepts_meta = ("dev",)  # batch[2] = ICBHI stethoscope id (data.params.meta_label_mode=dev)

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        # same swapped quirk as patch-mix (see ast_patchmix.py docstring)
        input_fdim: int = 798,
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: str | None = None,
        verbose: bool = False,
        # ---- sgscl training method (main.py arguments) ----
        method: str = "sgscl",          # "ce" | "dat" | "sgscl"
        m_cls: int = 4,                  # number of meta (dev) classes — DAT domain head dimension
        proj_dim: int = 768,
        temperature: float = 0.06,
        alpha: float = 1.0,
        target_type: str = "project1_project2block",
        grl_alpha: float = 1.0,          # the schedule saturates at 1.0 in the official recipe (methods/sgscl.py)
        cls_feat: str = "feat0",         # classifier input features[0]|[1] (feat1 = through GRL, experimental flag)
        con_feat: str = "feat0",         # sgscl contrastive features features[0]|[1] (feat1 = GitHub HEAD 2ed1bac)
        specaug_policy: str = "icbhi_ast_sup",
        specaug_mask: str = "mean",
        input_layout: str = "time_freq",
    ):
        # the parent (method="ce") builds the skeleton + mlp_head + classifier (=deepcopy(mlp_head)).
        super().__init__(
            n_cls=n_cls, fstride=fstride, tstride=tstride,
            input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain,
            model_size=model_size, pretrained_path=pretrained_path, verbose=verbose,
            method="ce", input_layout=input_layout,
        )
        if method not in {"ce", "dat", "sgscl"}:
            raise ValueError(f"method must be ce|dat|sgscl, got {method!r}")
        self.method = method
        self.alpha = alpha
        self.grl_alpha = grl_alpha
        self.target_type = target_type
        self.cls_feat = cls_feat
        self.con_feat = con_feat

        # original ast.py:72/138 — m_cls only for DAT; otherwise it stays at the default 527, unused.
        domain_label_dim = m_cls if method == "dat" else 527
        self.domain_mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, domain_label_dim),
        )

        if method == "dat":
            # set_model: domain_classifier = deepcopy(model.domain_mlp_head)
            self.domain_classifier = deepcopy(self.domain_mlp_head)
            self.metacl = None
        elif method == "sgscl":
            # set_model(domain_adaptation2): domain_classifier = Projector(768, proj_dim)
            self.domain_classifier = Projector(self.final_feat_dim, proj_dim)
            self.metacl = MetaCL(temperature=temperature)
        else:
            self.domain_classifier = None
            self.metacl = None

        # original train(): applies args.transforms = SpecAugment(args) to the batch (not in evaluation)
        self.specaug = SpecAugment(policy=specaug_policy, mask=specaug_mask)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        x = self._to_paper_layout(audio_batch)  # (B, 798, 128) = (B, time, freq)

        if not self.training:
            # original validate(): features→classifier[0]→criterion[0] without augmentation

            logits = self.classifier(self.forward_features(x))
            return logits, loss_func(logits, labels)

        return sgscl_training_losses(
            method=self.method,
            paper_features=self.forward_features,
            x_paper=x,
            specaug=self.specaug,
            classifier=self.classifier,
            domain_classifier=self.domain_classifier,
            metacl=self.metacl,
            loss_func=loss_func,
            labels=labels,
            meta_labels=meta_labels,
            alpha=self.alpha,
            grl_alpha=self.grl_alpha,
            target_type=self.target_type,
            cls_feat=self.cls_feat,
            con_feat=self.con_feat,
        )
