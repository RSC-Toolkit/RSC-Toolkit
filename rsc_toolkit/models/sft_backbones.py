# Portions of this file are ported from SG-SCL (https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Four SFT backbones of the patch-mix / SG-SCL family of papers — exact port of the official code.

Sources:
- Official code: https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning
  models/resnet.py, models/efficientnet.py, models/cnn.py (partly derived from PANN)
- Reported numbers: CE rows of Interspeech 2023 (patch-mix, https://arxiv.org/abs/2305.14032) Table 1 and
  ICASSP 2024 (SG-SCL, https://arxiv.org/abs/2312.09603) Table 3.

Kept as in the original code:
- ResNet/EfficientNet: torchvision model subclasses, first conv replaced with a 1-channel conv, fc/classifier removed.
  When loading ImageNet weights the first conv weights are **dropped and left randomly initialized** (original behavior).
- CNN6: 4 PANN ConvBlock5x5 stages + pooling (mean over freq axis), then (max+mean sum over time axis).
  Only overlapping keys are loaded from the AudioSet PANN ckpt (Cnn6_mAP=0.343).
- (args, alpha, training) in the forward signature and the DA branch (used by SG-SCL/DAT).

The wrappers (registered models) consist of backbone + nn.Linear(feat_dim, n_cls) classifier, same as the original set_model.
Toolkit datamodule features (B, n_mels, frames) are accepted with input_layout="freq_time"
and converted to the original input (B, 1, frames, n_mels).
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
    "pretrained_models/Cnn6_mAP=0.343.pth",  # location used by the original code
)


# ---------------------------------------------------------------- PANN blocks (original)
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


# ------------------------------------------------------------------ four backbones (original)
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
        """AudioSet-pretrained PANN CNN6 (https://zenodo.org/record/3960586) — loads only overlapping keys (original behavior)."""
        from rsc_toolkit.paths import download_target, resolve_pretrained

        ckpt = resolve_pretrained(PANN_CNN6_CANDIDATES)  # RSCTK_PRETRAINED_ROOT, then cwd
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

        x = torch.mean(x, dim=3)          # (the original comment says time, but it is actually the freq axis) — kept as in the original
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        if training:
            if args.domain_adaptation or args.domain_adaptation2:
                domain_x = ReverseLayerF.apply(x, alpha)
                x = (x, domain_x)
        return x


# ------------------------------------------------------------- registered wrappers (equivalent to set_model)
class _SFTClassifierModel(BaseRSCModel, nn.Module):
    """
    backbone + nn.Linear classifier (non-AST configuration of the original set_model).

    method:
    - "ce"    — SFT baseline. SpecAugment is applied once per batch, as in the original train loop.
    - "dat"   — domain adversarial training: nn.Linear(feat_dim, m_cls) domain head + GRL + unweighted CE.
    - "sgscl" — Projector(feat_dim, proj_dim) + MetaCL, two views (the second masks cumulatively on top of the first).
    Branch structure and losses are shared with rsc_toolkit/methods/sgscl.py (port of the original main.py train()).

    specaug_policy=None means no training augmentation — this default keeps the existing SFT goldens/smoke tests and the
    datamodule-side per-sample SpecAugment configuration unchanged. To reproduce the paper
    protocol, set the policy in the config and disable datamodule augmentation.
    """
    accepts_meta = ("dev",)  # batch[2] = ICBHI stethoscope id (sgscl_nonast data; method=ce accepts but ignores it)

    backbone_cls = None  # set by subclasses

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
            # original main.py:244 — non-AST uses nn.Linear(final_feat_dim, m_cls)
            self.domain_classifier = nn.Linear(self.backbone.final_feat_dim, m_cls)
            self.metacl = None
        elif method == "sgscl":
            # original main.py:247 — Projector(final_feat_dim, proj_dim)
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
            x = x.transpose(-2, -1)  # -> (B, 1, T, F) = original input
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._to_paper_layout(x))

    def _paper_features(self, x_paper: torch.Tensor) -> torch.Tensor:
        """(B, T, F) paper-layout 3D → features (for the shared methods/sgscl.py path)."""
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
