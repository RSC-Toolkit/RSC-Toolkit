"""
SG-SCL 논문(ICASSP 2024)의 AST — 공식 코드 충실 이식 (P0).

출처:
- 공식 코드: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
  models/ast.py + main.py set_model()/train().
- AST 골격·pos_embed 적응·forward 는 patch-mix 의 ast.py 와 문자 그대로 동일해서
  PatchMixAST 를 상속해 재사용한다 (골든으로 동일성 고정됨). sgscl 쪽 델타만 더한다:
  domain_mlp_head, 도메인 분류기/Projector, GRL, MetaCL, 배치 단위 SpecAugment 2회.

원본 그대로 유지하는 quirk:
- domain_mlp_head 차원: set_model 은 --domain_adaptation(DAT)일 때만
  domain_label_dim=m_cls 를 넘긴다. SG-SCL(domain_adaptation2)·CE 에서는 기본값
  527(AudioSet)로 만들어져 **한 번도 안 쓰인 채** state_dict 에 남는다.
- SpecAugment 는 train 루프에서 배치 전체에 적용된다(배치당 마스크 1벌). in-place
  구현이라 SG-SCL 의 두 번째 뷰는 첫 뷰 위에 마스크가 누적된다
  (methods/sgscl.py docstring 참고). 평가에는 적용하지 않는다.
- SG-SCL 에서 GRL(features[1]) 사용은 저장소 버전에 따라 다르다: 우리 골든 앵커인
  클론 6656460(2024-12-05)은 대조 특징에 features[0](GRL 미사용)을 쓰고, 이후 GitHub
  HEAD 커밋 2ed1bac 은 features[1](GRL 통과)로 바꿨다. 기본값 con_feat="feat0" 은
  클론(=기존 P1) 동작이고, con_feat="feat1" 이 GitHub HEAD, cls_feat="feat1" 은
  분류기 입력까지 GRL 로 바꾸는 실험 플래그(2026-08-28 사용자 지시)다. GRL 은
  파라미터가 없어 어느 조합이든 state_dict·평가 경로는 동일하다(기존 ckpt 호환).
- 원본의 미사용 projector(생성만 되고 MA 로 스냅샷과 섞이는 것)는 재현하지 않는다.
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
    """원본 sgscl ASTModel + set_model 구성 (method: ce | dat | sgscl)."""

    accepts_meta = ("dev",)  # batch[2] = ICBHI 청진기 id (data.params.meta_label_mode=dev)

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        # patch-mix 와 같은 뒤집힌 quirk (ast_patchmix.py docstring)
        input_fdim: int = 798,
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: str | None = None,
        verbose: bool = False,
        # ---- sgscl 학습 방법 (main.py 인자) ----
        method: str = "sgscl",          # "ce" | "dat" | "sgscl"
        m_cls: int = 4,                  # meta(dev) 클래스 수 — DAT 도메인 헤드 차원
        proj_dim: int = 768,
        temperature: float = 0.06,
        alpha: float = 1.0,
        target_type: str = "project1_project2block",
        grl_alpha: float = 1.0,          # 공식 레시피에서 스케줄이 1.0 으로 포화 (methods/sgscl.py)
        cls_feat: str = "feat0",         # 분류기 입력 features[0]|[1] (feat1 = GRL 통과, 2026-08-28 실험 플래그)
        con_feat: str = "feat0",         # sgscl 대조 특징 features[0]|[1] (feat1 = GitHub HEAD 2ed1bac)
        specaug_policy: str = "icbhi_ast_sup",
        specaug_mask: str = "mean",
        input_layout: str = "time_freq",
    ):
        # 부모(method="ce")가 골격 + mlp_head + classifier(=deepcopy(mlp_head)) 를 만든다.
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

        # 원본 ast.py:72/138 — DAT 일 때만 m_cls, 그 외엔 기본 527 인 채 미사용 잔존.
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

        # 원본 train(): args.transforms = SpecAugment(args) 를 배치에 적용 (평가 제외)
        self.specaug = SpecAugment(policy=specaug_policy, mask=specaug_mask)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        x = self._to_paper_layout(audio_batch)  # (B, 798, 128) = (B, time, freq)

        if not self.training:
            # 원본 validate(): 증강 없이 features→classifier[0]→criterion[0]
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
