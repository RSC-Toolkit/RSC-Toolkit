"""
Patch-Mix 논문(Interspeech 2023)의 AST 모델 — 공식 코드 충실 이식 (P0).

출처:
- 논문: Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on
  Respiratory Sound Classification (https://arxiv.org/abs/2305.14032)
- 공식 코드: https://github.com/raymin0223/patch-mix_contrastive_learning
  models/ast.py (MIT, Copyright (c) Sangmin Bae; 원류는 YuanGongND/ast)
- 같은 구조를 kaen2891/sgscl(ICASSP 2024), wa976/PAFA 등 후속 저장소들이 공유한다.

이 모듈은 P0 원칙(구조·초기화·forward 를 공식 코드와 동일하게)을 따른다.
공식 코드의 "버그처럼 보이는 동작"도 그대로 유지한다. 특히:

**뒤집힌 입력 차원 quirk (그대로 유지할 것):**
원본 main.py:242-243 은 (h,w)=(798,128) 스펙트로그램에 대해 input_fdim=798,
input_tdim=128 로 — 즉 주파수/시간이 뒤집힌 값으로 — 모델을 만든다. 이 때문에
AudioSet pos_embed(12x101 격자)가 t_dim=12 로 잘리고 f_dim=79 로 보간되는데,
forward 의 transpose(2,3) 뒤 실제 conv 패치 격자는 freq 12 x time 79 라서
pos_embed 의 배열이 원 AST 의 의도와 어긋난 채로 미세조정된다.
논문 수치는 이 상태로 학습된 결과이므로 여기서는 이를 그대로 재현한다.
(기본값이 곧 논문 ICBHI 설정: input_fdim=798, input_tdim=128)

일반 용도의 "올바른" AST 는 rsc_toolkit.models.ast.ASTClassifier 를 쓴다.

우리 툴킷과의 비구조적 차이 (가중치·수치에 영향 없음):
- timm 0.4.5 대신 툴킷 핀 버전(1.x)의 DeiT-distilled 골격을 쓴다. 파라미터
  키 이름·모양은 동일함을 확인했고, 골든 테스트가 이를 고정한다.
- 사전학습 ckpt 는 pretrained_models/upstream/ 의 아카이브(출처 기록 포함)를
  기본으로 쓰고, 없으면 원 Dropbox URL 에서 받는다.
- forward 의 @autocast 데코레이터는 없다. AMP 는 학습 루프에서 적용한다
  (원본도 train/validate 에서 autocast 로 감싼다).
- 원본 set_model(main.py:259)의 classifier = deepcopy(model.mlp_head) 를
  생성자 안에서 수행해 self.classifier 로 둔다. mlp_head 는 원본과 같이
  forward 에서 쓰이지 않는 채로 남는다(원본 optimizer 에도 포함되지만
  gradient 를 받지 않는 파라미터다).
"""
from __future__ import annotations

import logging
import os
import pathlib
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .projector import Projector

try:  # timm 은 models extra 의존성
    import timm
    from timm.models.layers import to_2tuple, trunc_normal_
except Exception as e:  # pragma: no cover
    timm = None
    to_2tuple = None
    trunc_normal_ = None
    _timm_import_error = e

logger = logging.getLogger(__name__)

AUDIOSET_CKPT_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
# 아카이브된 ckpt 후보 (프로젝트 루트 기준). 순서대로 찾는다.
AUDIOSET_CKPT_CANDIDATES = (
    "pretrained_models/upstream/audioset_0.4593.pth",
    "pretrained_models/audioset_10_10_0.4593.pth",  # 원본 코드가 쓰는 파일명
)


class PatchEmbed(nn.Module):
    """원본과 동일한 timm PatchEmbed 대체 — 입력 크기 제약을 없앤 conv 패치화."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def _create_deit_skeleton(model_size: str, pretrained: bool):
    """timm DeiT-distilled 골격 생성 (0.4.5 이름 → 1.x 이름 폴백)."""
    names = {
        "tiny224": ("vit_deit_tiny_distilled_patch16_224", "deit_tiny_distilled_patch16_224"),
        "small224": ("vit_deit_small_distilled_patch16_224", "deit_small_distilled_patch16_224"),
        "base224": ("vit_deit_base_distilled_patch16_224", "deit_base_distilled_patch16_224"),
        "base384": ("vit_deit_base_distilled_patch16_384", "deit_base_distilled_patch16_384"),
    }
    if model_size not in names:
        raise ValueError("model_size must be one of tiny224, small224, base224, base384")
    old, new = names[model_size]
    try:
        return timm.create_model(old, pretrained=pretrained)
    except (RuntimeError, Exception) as exc:  # timm 1.x 는 옛 이름을 모른다
        if "Unknown model" not in str(exc):
            raise
        return timm.create_model(new, pretrained=pretrained)


def _resolve_audioset_ckpt(pretrained_path: Optional[str]) -> pathlib.Path:
    if pretrained_path is not None:
        return pathlib.Path(pretrained_path).expanduser()
    for cand in AUDIOSET_CKPT_CANDIDATES:
        p = pathlib.Path(cand)
        if p.exists():
            return p
    # 없으면 원 출처에서 받는다 (다운로드 출처 기록 포함)
    from rsc_toolkit.datasets.download_utils import download_file, record_download_provenance

    target = pathlib.Path(AUDIOSET_CKPT_CANDIDATES[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("AudioSet AST ckpt 가 없어 내려받는다: %s", AUDIOSET_CKPT_URL)
    download_file(AUDIOSET_CKPT_URL, target)
    record_download_provenance(target, url=AUDIOSET_CKPT_URL)
    return target


@ModelFactory.register_model_class("ast_patchmix")
class PatchMixAST(BaseRSCModel, nn.Module):
    """
    원본 ASTModel + set_model 의 classifier 구성을 합친 논문 충실 모델.

    forward(x) 는 분류 logits 를 돌려준다 (x: [B, 1, 798, 128] 또는 [B, 798, 128]).
    학습 방법(patchmix / patchmix_cl)은 forward_features(..., patch_mix=True) 를 쓴다.
    """

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        # 주의: 아래 두 기본값은 일부러 뒤집혀 있다 (모듈 docstring 의 quirk 참고).
        input_fdim: int = 798,
        input_tdim: int = 128,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        mix_beta: float = 1.0,
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        # ---- 학습 방법 (원본 main.py 의 --method 와 관련 인자들) ----
        method: str = "ce",              # "ce" | "patchmix" | "patchmix_cl"
        proj_dim: int = 768,
        temperature: float = 0.06,
        alpha: float = 1.0,
        negative_pair: str = "all",
        target_type: str = "grad_block",
        time_domain: bool = False,
        # 툴킷 datamodule 은 (B, n_mels, frames) 로 주므로 "freq_time".
        # 원본과 같은 (B, frames, n_mels) 직접 입력이면 "time_freq" (기본, 충실).
        input_layout: str = "time_freq",
    ):
        super().__init__()
        if timm is None:  # pragma: no cover
            raise ImportError("PatchMixAST requires timm (`pip install rsc-toolkit[models]`).") from _timm_import_error

        self.final_feat_dim = 768
        self.mix_beta = mix_beta

        if audioset_pretrain and not imagenet_pretrain:
            # 원본과 동일한 제약 (ast.py:114-115)
            raise ValueError("audioset-only pretraining is not supported; set imagenet_pretrain=True")

        if not audioset_pretrain:
            self._build_imagenet_or_scratch(
                n_cls, fstride, tstride, input_fdim, input_tdim, imagenet_pretrain, model_size, verbose
            )
        else:
            if model_size != "base384":
                raise ValueError("currently only has base384 AudioSet pretrained model.")
            self._build_audioset(n_cls, fstride, tstride, input_fdim, input_tdim, pretrained_path, verbose)

        # 원본 main.py:259 — classifier 는 (새로 초기화된) mlp_head 의 deepcopy.
        self.classifier = deepcopy(self.mlp_head)

        if method not in {"ce", "patchmix", "patchmix_cl"}:
            raise ValueError(f"method must be ce|patchmix|patchmix_cl, got {method!r}")
        if input_layout not in {"time_freq", "freq_time"}:
            raise ValueError(f"input_layout must be time_freq|freq_time, got {input_layout!r}")
        self.method = method
        self.alpha = alpha
        self.target_type = target_type
        self.time_domain = time_domain
        self.input_layout = input_layout
        # 원본 set_model: patchmix_cl 일 때만 Projector, 아니면 Identity (optimizer 포함 동일)
        if method == "patchmix_cl":
            self.projector = Projector(self.final_feat_dim, proj_dim)
            from rsc_toolkit.methods.patchmix import PatchMixConLoss

            self.patchmix_con_loss = PatchMixConLoss(temperature=temperature, negative_pair=negative_pair)
        else:
            self.projector = nn.Identity()

    # ------------------------------------------------------------------ 구성
    def _build_imagenet_or_scratch(self, n_cls, fstride, tstride, input_fdim, input_tdim,
                                   imagenet_pretrain, model_size, verbose):
        """원본 ast.py:56-110 (audioset_pretrain == False 분기)과 동일."""
        self.v = _create_deit_skeleton(model_size, pretrained=imagenet_pretrain)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)  # 원본의 오타(oringal)까지 유지
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls)
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self._replace_patch_embed(input_fdim, input_tdim, fstride, tstride, num_patches,
                                  copy_imagenet_proj=imagenet_pretrain)
        if verbose:
            logger.info("PatchMixAST(non-audioset): f_dim=%d t_dim=%d patches=%d", f_dim, t_dim, num_patches)

        if imagenet_pretrain:
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            )
            if t_dim <= self.oringal_hw:
                start = int(self.oringal_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                )
            if f_dim <= self.oringal_hw:
                start = int(self.oringal_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            new_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 2, self.original_embedding_dim)
            )
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)

    def _build_audioset(self, n_cls, fstride, tstride, input_fdim, input_tdim, pretrained_path, verbose):
        """
        원본 ast.py:113-157 (audioset_pretrain == True 분기)과 동일한 결과를 만든다.

        원본은 내부 ASTModel(128,1024, imagenet=False)을 만들어 DataParallel 로 감싸
        ckpt 를 통째로 로드한 뒤 .v 만 취한다. 여기서는 같은 결과를 직접 구성한다:
        (1) 골격 + 1채널 conv 패치화, (2) pos_embed 를 ckpt 모양(1, 1214, 768)으로
        재설정, (3) ckpt 의 module.v.* 가중치 로드, (4) 12x101 격자 기준 절단/보간.
        """
        self.v = _create_deit_skeleton("base384", pretrained=False)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        # 내부 모델(128x1024, fstride/tstride 10)의 패치 수 = 12*101 = 1212
        inner_patches = 12 * 101
        self._replace_patch_embed(input_fdim, input_tdim, fstride, tstride, inner_patches,
                                  copy_imagenet_proj=False)
        self.v.pos_embed = nn.Parameter(
            torch.zeros(1, inner_patches + 2, self.original_embedding_dim)
        )

        ckpt_path = _resolve_audioset_ckpt(pretrained_path)
        sd = torch.load(ckpt_path, map_location="cpu")
        v_state = {}
        for key, value in sd.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("v."):
                v_state[key[len("v."):]] = value
            # module.mlp_head.* (527-클래스 AudioSet 헤드)는 원본에서도 outer 가 버린다
        missing, unexpected = self.v.load_state_dict(v_state, strict=False)
        # head/head_dist 는 ckpt·골격 어느 쪽에도 있을 수 있는 미사용 분류 헤드다
        real_missing = [k for k in missing if not k.startswith(("head", "head_dist"))]
        real_unexpected = [k for k in unexpected if not k.startswith(("head", "head_dist"))]
        if real_missing or real_unexpected:
            raise RuntimeError(
                f"AudioSet ckpt 로드가 원본과 다르게 됐다. missing={real_missing[:5]} unexpected={real_unexpected[:5]}"
            )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls)
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose:
            logger.info("PatchMixAST(audioset): f_dim=%d t_dim=%d patches=%d", f_dim, t_dim, num_patches)

        # 원본 ast.py:144-157 그대로 (상수 1212/12/101, 절단 기준 50/6 포함)
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
        if t_dim < 101:
            new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")
        if f_dim < 12:
            new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2) : 6 - int(f_dim / 2) + f_dim, :]
        elif f_dim > 12:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
        new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def _replace_patch_embed(self, input_fdim, input_tdim, fstride, tstride, num_patches, copy_imagenet_proj):
        """timm 골격의 patch_embed 를 원본과 같은 제약 없는 PatchEmbed 로 교체."""
        new_patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim), patch_size=16, in_chans=1,
            embed_dim=self.original_embedding_dim,
        )
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if copy_imagenet_proj:
            # 원본 ast.py:83-84 — RGB 3채널 conv 가중치를 채널 합으로 1채널화
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        new_patch_embed.proj = new_proj
        new_patch_embed.num_patches = num_patches
        self.v.patch_embed = new_patch_embed

    # ------------------------------------------------------------- 원본 메서드
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024) -> Tuple[int, int]:
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        return test_out.shape[2], test_out.shape[3]

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        return patch.reshape(B, h, w, dim)

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        return square.reshape(B, h * w, dim)

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        """원본 ast.py:178-208 과 동일. lam 은 실제 교체 비율로 재계산된다."""
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        index = torch.randperm(batch_size).to(device)

        if not time_domain:
            num_mask = int(num_patch * (1.0 - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1.0 - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)

        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    def forward_features(self, x, y=None, patch_mix=False, time_domain=False):
        """원본 ASTModel.forward 와 동일 (autocast 데코레이터만 제외 — 모듈 docstring)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index

    def _to_paper_layout(self, x: torch.Tensor) -> torch.Tensor:
        """툴킷 피처(freq_time: B,128,798)를 원본 입력(time_freq: B,798,128)으로."""
        if self.input_layout == "freq_time":
            return x.transpose(-2, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(self._to_paper_layout(x)))

    def training_forward(self, audio_batch, labels, loss_func):
        """
        원본 main.py train() 의 method 분기와 동일. 평가 모드(validate)는 원본처럼
        항상 ce 경로다 (원본 validate 는 features→classifier→criterion[0]).

        loss_func 은 build_loss 산물이다: ce 방법은 weighted CE
        (--weighted_loss), patchmix/patchmix_cl 은 비가중 CE (스크립트와 동일).
        """
        x = self._to_paper_layout(audio_batch)

        if (not self.training) or self.method == "ce":
            feats = self.forward_features(x)
            logits = self.classifier(feats)
            return logits, loss_func(logits, labels)

        if self.method == "patchmix":
            from rsc_toolkit.methods.patchmix import PatchMixLoss

            mix_feats, y_a, y_b, lam, index = self.forward_features(
                x, y=labels, patch_mix=True, time_domain=self.time_domain
            )
            logits = self.classifier(mix_feats)
            loss = PatchMixLoss(criterion=loss_func)(logits, y_a, y_b, lam)
            return logits, loss

        # patchmix_cl (원본 main.py:355-372)
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        loss = loss_func(logits, labels)

        if self.target_type == "grad_block":
            proj1 = feats.detach().clone()
        elif self.target_type == "grad_flow":
            proj1 = feats
        elif self.target_type == "project_block":
            proj1 = self.projector(feats).detach().clone()
        elif self.target_type == "project_flow":
            proj1 = self.projector(feats)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type!r}")

        mix_feats, y_a, y_b, lam, index = self.forward_features(
            x, y=labels, patch_mix=True, time_domain=self.time_domain
        )
        proj2 = self.projector(mix_feats)
        loss = loss + self.alpha * self.patchmix_con_loss(proj1, proj2, labels, y_b, lam, index)
        return logits, loss
