# Portions of this file are ported from SG-SCL (https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch.hub import load_state_dict_from_url


def _make_norm(track_bn: bool):
    def norm_layer(*args, **kwargs):
        return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)

    return norm_layer


class ResNet10(torchvision.models.resnet.ResNet):
    final_feat_dim = 512

    def __init__(self, track_bn: bool = True):
        super().__init__(torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], norm_layer=_make_norm(track_bn))
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official supervised weights for ResNet10.")

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for ResNet10.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet18(torchvision.models.resnet.ResNet):
    final_feat_dim = 512

    def __init__(self, track_bn: bool = True):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], norm_layer=_make_norm(track_bn))
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1.url, progress=progress
        )
        del state_dict["conv1.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for ResNet18.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet34(torchvision.models.resnet.ResNet):
    final_feat_dim = 512

    def __init__(self, track_bn: bool = True):
        super().__init__(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], norm_layer=_make_norm(track_bn))
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1.url, progress=progress
        )
        del state_dict["conv1.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for ResNet34.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet50(torchvision.models.resnet.ResNet):
    final_feat_dim = 2048

    def __init__(self, track_bn: bool = True):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], norm_layer=_make_norm(track_bn))
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1.url, progress=progress
        )
        del state_dict["conv1.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        try:
            from pl_bolts.models.self_supervised import SimCLR
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SimCLR weights require lightning-bolts (pl_bolts).") from exc

        weight_path = (
            "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/"
            "simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
        )
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

        state_dict = {}
        for k, v in simclr.state_dict().items():
            if "encoder." in k:
                k = k.replace("encoder.", "")
            if "fc" not in k or "project" not in k:
                state_dict[k] = v
        missing, _ = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            raise AssertionError("Model code may be incorrect")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet101(torchvision.models.resnet.ResNet):
    final_feat_dim = 2048

    def __init__(self, track_bn: bool = True):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], norm_layer=_make_norm(track_bn))
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            torchvision.models.ResNet101_Weights.IMAGENET1K_V1.url, progress=progress
        )
        del state_dict["conv1.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for ResNet101.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)
