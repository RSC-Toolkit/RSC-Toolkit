# Portions of this file are ported from SG-SCL (https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.efficientnet import _efficientnet_conf


class EfficientNetB0(torchvision.models.efficientnet.EfficientNet):
    final_feat_dim = 1280

    def __init__(self):
        inverted_residual_setting, _ = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
        super().__init__(inverted_residual_setting, 0.2)
        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
            progress=progress,
        )
        del state_dict["features.0.0.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for EfficientNetB0.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class EfficientNetB1(torchvision.models.efficientnet.EfficientNet):
    final_feat_dim = 1280

    def __init__(self):
        inverted_residual_setting, _ = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
        super().__init__(inverted_residual_setting, 0.2)
        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
            progress=progress,
        )
        del state_dict["features.0.0.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for EfficientNetB1.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class EfficientNetB2(torchvision.models.efficientnet.EfficientNet):
    final_feat_dim = 1408

    def __init__(self):
        inverted_residual_setting, _ = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
        super().__init__(inverted_residual_setting, 0.3)
        del self.classifier
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def load_sl_official_weights(self, progress: bool = True) -> None:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
            progress=progress,
        )
        del state_dict["features.0.0.weight"]
        self.load_state_dict(state_dict, strict=False)

    def load_ssl_official_weights(self, progress: bool = True) -> None:
        raise NotImplementedError("No official self-supervised weights for EfficientNetB2.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
