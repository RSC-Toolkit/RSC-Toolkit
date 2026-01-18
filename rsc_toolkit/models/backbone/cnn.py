from __future__ import annotations

from typing import Optional
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(layer: nn.Module) -> None:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(bn: nn.BatchNorm2d) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=stride,
            padding=(2, 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self._init_weight()

    def _init_weight(self) -> None:
        _init_layer(self.conv1)
        _init_bn(self.bn1)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type: str = "avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._init_weight()

    def _init_weight(self) -> None:
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type: str = "avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        return x


class CNN6(nn.Module):
    final_feat_dim = 512

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.do_dropout = dropout > 0
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64, stride=(1, 1))
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128, stride=(1, 1))
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256, stride=(1, 1))
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512, stride=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2


class CNN10(nn.Module):
    final_feat_dim = 512

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.do_dropout = dropout > 0
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2


class CNN14(nn.Module):
    final_feat_dim = 2048

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.do_dropout = dropout > 0
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2


def load_cnn_checkpoint(model: nn.Module, checkpoint_path: str | pathlib.Path) -> None:
    checkpoint_path = pathlib.Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Expected a state_dict or a dict with key 'model'")
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()}, strict=False)
