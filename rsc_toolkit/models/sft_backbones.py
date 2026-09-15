"""
patch-mix / SG-SCL 계열 논문의 SFT 백본 4종 — 공식 코드 충실 이식 (P0).

출처:
- 공식 코드: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
  models/resnet.py, models/efficientnet.py, models/cnn.py (원류 일부는 PANN)
- 수치 출처: Interspeech 2023 (patch-mix, https://arxiv.org/abs/2305.14032) Table 1 및
  ICASSP 2024 (SG-SCL, https://arxiv.org/abs/2312.09603) Table 3 의 CE 행.

원본 그대로 유지한 것:
- ResNet/EfficientNet: torchvision 모델 서브클래스, 첫 conv 를 1채널로 교체, fc/classifier 삭제.
  ImageNet 가중치 로드 시 첫 conv 가중치는 **삭제하고 랜덤 초기화로 남긴다** (원본 동작).
- CNN6: PANN ConvBlock5x5 4단 + (freq 축 mean) 후 (time 축 max+mean 합) 풀링.
  AudioSet PANN ckpt(Cnn6_mAP=0.343)에서 겹치는 키만 로드.
- forward 시그니처의 (args, alpha, training) 와 DA 분기 (SG-SCL/DAT 에서 사용).

래퍼(등록 모델)는 원본 set_model 과 동일하게 backbone + nn.Linear(feat_dim, n_cls)
분류기로 구성한다. 툴킷 datamodule 피처 (B, n_mels, frames) 는 input_layout="freq_time"
으로 받아 원본 입력 (B, 1, frames, n_mels) 로 바꾼다.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models.efficientnet import _efficientnet_conf

from .base import BaseRSCModel
from .factory import ModelFactory
from .grl import ReverseLayerF

logger = logging.getLogger(__name__)

PANN_CNN6_URL = "https://zenodo.org/record/3960586/files/Cnn6_mAP%3D0.343.pth?download=1"
PANN_CNN6_CANDIDATES = (
    "pretrained_models/upstream/Cnn6_mAP=0.343.pth",
    "pretrained_models/Cnn6_mAP=0.343.pth",  # 원본 코드가 쓰는 위치
)


# ---------------------------------------------------------------- PANN 블록 (원문)
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock5x5(nn.Module):  # for CNN6
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(5, 5), stride=stride, padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")
        return x


# ------------------------------------------------------------------ 백본 4종 (원문)
class ResNet18SFT(torchvision.models.resnet.ResNet):
    def __init__(self, track_bn=True):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_feat_dim = 512

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.ResNet18_Weights.IMAGENET1K_V1.url, progress=progress)
        del state_dict["conv1.weight"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

    def forward(self, x, args=None, alpha=None, training=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if training:
            if args.domain_adaptation or args.domain_adaptation2:
                domain_x = ReverseLayerF.apply(x, alpha)
                x = (x, domain_x)
        return x


class ResNet34SFT(torchvision.models.resnet.ResNet):
    def __init__(self, track_bn=True):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer)
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_feat_dim = 512

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.ResNet34_Weights.IMAGENET1K_V1.url, progress=progress)
        del state_dict["conv1.weight"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

    forward = ResNet18SFT.forward


class EfficientNetB0SFT(torchvision.models.efficientnet.EfficientNet):
    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
        super().__init__(inverted_residual_setting, 0.2)

        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.final_feat_dim = 1280

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth", progress=progress
        )
        del state_dict["features.0.0.weight"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

    def forward(self, x, args=None, alpha=None, training=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if training:
            if args.domain_adaptation or args.domain_adaptation2:
                domain_x = ReverseLayerF.apply(x, alpha)
                x = (x, domain_x)
        return x


class CNN6SFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_feat_dim = 512

        self.do_dropout = False
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64, stride=(1, 1))
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128, stride=(1, 1))
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256, stride=(1, 1))
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512, stride=(1, 1))
        self.dropout = nn.Dropout(0.2)

    def load_sl_official_weights(self):
        """AudioSet 사전학습 PANN CNN6 (https://zenodo.org/record/3960586) — 겹치는 키만 로드 (원본 동작)."""
        from rsc_toolkit.paths import download_target, resolve_pretrained

        ckpt = resolve_pretrained(PANN_CNN6_CANDIDATES)  # RSCTK_PRETRAINED_ROOT → cwd 순
        if ckpt is None:
            from rsc_toolkit.datasets.download_utils import download_file, record_download_provenance

            ckpt = download_target(PANN_CNN6_CANDIDATES[0])
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            download_file(PANN_CNN6_URL, ckpt)
            record_download_provenance(ckpt, url=PANN_CNN6_URL)

        weights = torch.load(ckpt, map_location="cpu")["model"]
        state_dict = {k: v for k, v in weights.items() if k in self.state_dict().keys()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

    def forward(self, x, args=None, alpha=None, training=False):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3)          # (원본 주석은 time 이라 하나 실제는 freq 축) — 원문 그대로
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        if training:
            if args.domain_adaptation or args.domain_adaptation2:
                domain_x = ReverseLayerF.apply(x, alpha)
                x = (x, domain_x)
        return x


# ------------------------------------------------------------- 등록 래퍼 (set_model 상당)
class _SFTClassifierModel(BaseRSCModel, nn.Module):
    """
    backbone + nn.Linear 분류기 (원본 set_model 의 비-AST 구성).

    method:
    - "ce"    — SFT 기준선. SpecAugment 는 원본 train 루프처럼 배치 단위 1회.
    - "dat"   — 도메인 적대 학습: nn.Linear(feat_dim, m_cls) 도메인 헤드 + GRL + 비가중 CE.
    - "sgscl" — Projector(feat_dim, proj_dim) + MetaCL, 두 뷰(두 번째는 첫 뷰 위 누적 마스킹).
    분기 구성과 손실은 rsc_toolkit/methods/sgscl.py (원본 main.py train() 이식) 공용.

    specaug_policy=None 이면 학습 증강 없음 — 기존 SFT 골든/스모크와 datamodule 쪽
    per-sample SpecAugment 구성을 그대로 유지하기 위한 기본값이다. 논문 프로토콜
    재현 시에는 config 에서 policy 를 주고 datamodule 증강을 끈다.
    """
    accepts_meta = ("dev",)  # batch[2] = ICBHI 청진기 id (sgscl_nonast 데이터; method=ce 는 받되 무시)

    backbone_cls = None  # 서브클래스에서 지정

    def __init__(
        self,
        n_cls: int = 4,
        from_sl_official: bool = True,
        input_layout: str = "freq_time",
        method: str = "ce",
        m_cls: int = 4,
        proj_dim: int = 768,
        temperature: float = 0.06,
        alpha: float = 1.0,
        target_type: str = "project1_project2block",
        grl_alpha: float = 1.0,
        specaug_policy: Optional[str] = None,
        specaug_mask: str = "mean",
    ):
        super().__init__()
        if input_layout not in {"time_freq", "freq_time"}:
            raise ValueError(f"input_layout must be time_freq|freq_time, got {input_layout!r}")
        if method not in {"ce", "dat", "sgscl"}:
            raise ValueError(f"method must be ce|dat|sgscl, got {method!r}")
        self.backbone = self.backbone_cls()
        self.classifier = nn.Linear(self.backbone.final_feat_dim, n_cls)
        self.input_layout = input_layout
        self.method = method
        self.alpha = alpha
        self.grl_alpha = grl_alpha
        self.target_type = target_type
        if from_sl_official:
            self.backbone.load_sl_official_weights()

        from rsc_toolkit.methods.sgscl import MetaCL
        from rsc_toolkit.models.projector import Projector

        if method == "dat":
            # 원본 main.py:244 — 비-AST 는 nn.Linear(final_feat_dim, m_cls)
            self.domain_classifier = nn.Linear(self.backbone.final_feat_dim, m_cls)
            self.metacl = None
        elif method == "sgscl":
            # 원본 main.py:247 — Projector(final_feat_dim, proj_dim)
            self.domain_classifier = Projector(self.backbone.final_feat_dim, proj_dim)
            self.metacl = MetaCL(temperature=temperature)
        else:
            self.domain_classifier = None
            self.metacl = None

        if specaug_policy is not None:
            from rsc_toolkit.augmentations.specaugment import SpecAugment

            self.specaug = SpecAugment(policy=specaug_policy, mask=specaug_mask)
        else:
            self.specaug = None

    def _to_paper_layout(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, F, T) -> (B, 1, F, T)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)  # -> (B, 1, T, F) = 원본 입력
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._to_paper_layout(x))

    def _paper_features(self, x_paper: torch.Tensor) -> torch.Tensor:
        """(B, T, F) 페이퍼 레이아웃 3D → 특징 (methods/sgscl.py 공용 경로용)."""
        return self.backbone(x_paper.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if not self.training:
            logits = self.forward(audio_batch)
            return logits, loss_func(logits, labels)

        from rsc_toolkit.methods.sgscl import sgscl_training_losses

        x = audio_batch
        if x.dim() == 4:
            x = x.squeeze(1)
        x_paper = x.transpose(-2, -1) if self.input_layout == "freq_time" else x  # (B, T, F)
        return sgscl_training_losses(
            method=self.method,
            paper_features=self._paper_features,
            x_paper=x_paper,
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
        )


@ModelFactory.register_model_class("resnet18_sft")
class ResNet18SFTClassifier(_SFTClassifierModel):
    backbone_cls = ResNet18SFT


@ModelFactory.register_model_class("resnet34_sft")
class ResNet34SFTClassifier(_SFTClassifierModel):
    backbone_cls = ResNet34SFT


@ModelFactory.register_model_class("efficientnet_b0_sft")
class EfficientNetB0SFTClassifier(_SFTClassifierModel):
    backbone_cls = EfficientNetB0SFT


@ModelFactory.register_model_class("cnn6_sft")
class CNN6SFTClassifier(_SFTClassifierModel):
    backbone_cls = CNN6SFT
