from __future__ import annotations

import pytest
import torch

from rsc_toolkit.models.cnn import CNNClassifier
from rsc_toolkit.models.efficientnet import EfficientNetClassifier
from rsc_toolkit.models.resnet import ResNetClassifier


@pytest.mark.slow
@pytest.mark.parametrize("backbone", ["cnn6", "cnn10", "cnn14"])
def test_cnn_classifier_forward(backbone: str) -> None:
    model = CNNClassifier(n_cls=4, backbone=backbone, dropout=0.0)
    x = torch.randn(2, 64, 64)
    y = model(x)
    assert y.shape == (2, 4)


@pytest.mark.slow
@pytest.mark.parametrize("backbone", ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101"])
def test_resnet_classifier_forward(backbone: str) -> None:
    model = ResNetClassifier(n_cls=4, backbone=backbone, pretrained=None, track_bn=True)
    x = torch.randn(2, 64, 64)
    y = model(x)
    assert y.shape == (2, 4)


@pytest.mark.slow
@pytest.mark.parametrize("backbone", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"])
def test_efficientnet_classifier_forward(backbone: str) -> None:
    model = EfficientNetClassifier(n_cls=4, backbone=backbone, pretrained=None)
    x = torch.randn(2, 64, 64)
    y = model(x)
    assert y.shape == (2, 4)


@pytest.mark.slow
def test_ast_classifier_forward() -> None:
    from rsc_toolkit.models.ast import ASTClassifier

    model = ASTClassifier(
        n_cls=4,
        fstride=16,
        tstride=16,
        input_fdim=224,
        input_tdim=224,
        pretrained="none",
        pretrained_path=None,
        download_if_missing=False,
        model_size="tiny224",
        dropout=0.0,
    )
    x = torch.randn(2, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
