"""
Classifier of Domain Transfer (Wang & Wang, ICASSP 2022) — **re-implementation from the paper description** (no official or unofficial code, grade C).

Source: "A Domain Transfer Based Data Augmentation Method for Automated Respiratory Classification",
ICASSP 2022 pp.9017–9020, DOI 10.1109/ICASSP43922.2022.9746941 (§2.2 NetWork Structure).

Architecture stated in the paper (last sentence of §2.2, verbatim):
  backbone(OutDim 2048) → GeMPool(frequency axis) → GeMPool(time axis) → BatchNorm → Dropout
  → Linear(2048→512) → ReLU → Dropout → Linear(512→512) → ReLU → Dropout → Linear(512→4)
- The backbone is an **ImageNet-pretrained ResNeSt + IBN modules**. All dropout rates are 0.5.
- GeMPool: F = (mean(x^p))^(1/p), p is a learnable parameter (paper Eq. 2). p=1 gives average pooling, p→∞ max pooling.

Assumptions (not specified in the paper — all exposed as arguments):
- **ResNeSt variant**: the paper only says "ResNeSt" without the depth. An output of 2048 fits 50d/101e/200e/269e alike.
  The default is `resnest50d` (most standard and lightest); it can be changed with the `backbone` argument.
- **IBN type and position**: the paper only says "IBN module". The default configuration of the original IBN-Net paper, **IBN-a applied to bn1 in stages 1–3**
  (`ibn_cfg=('a','a','a',None)`, IBN-Net `resnet_ibn.py:115`), is adopted. **No public ResNeSt-IBN ImageNet weights
  exist** (IBN-Net only releases ResNet/ResNeXt/SE-ResNet/DenseNet; timm has no IBN) → timm ResNeSt ImageNet weights are
  loaded, then bn1 is replaced by IBN, which inherits the affine parameters of the original BN split as first half=IN / second half=BN
  (IN does not use running stats, so the statistics of that half are discarded).
- **Input channels**: the ImageNet backbone has 3 channels but the spectrogram has 1. Following the AST/DASS convention, the first conv weights
  are summed over the channel axis to fold into 1 channel (`torch.sum(w, dim=1, keepdim=True)`).
- **Spectrogram type**: the paper only says "STFT with 512 FFT, 256 hop" and does not say whether mel is used. The Fig.1 caption says
  "log-spectrogram", so the **log of the linear STFT magnitude** is the default; `use_mel=True` selects log-mel.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsc_toolkit.augmentations.domain_transfer import mixup_batch

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)


class IBN(nn.Module):
    """IBN layer from IBN-Net (arXiv 1807.09441) `ibnnet/modules.py` — half of the channels IN / half BN."""

    def __init__(self, planes: int, ratio: float = 0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, 1)
        return torch.cat((self.IN(split[0].contiguous()), self.BN(split[1].contiguous())), 1)

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm2d) -> "IBN":
        """Inherits the affine of a pretrained BatchNorm as first half=IN / second half=BN (IN has no running stats)."""
        m = cls(bn.num_features)
        h = m.half
        with torch.no_grad():
            if bn.affine:
                m.IN.weight.copy_(bn.weight[:h]); m.IN.bias.copy_(bn.bias[:h])
                m.BN.weight.copy_(bn.weight[h:]); m.BN.bias.copy_(bn.bias[h:])
            m.BN.running_mean.copy_(bn.running_mean[h:])
            m.BN.running_var.copy_(bn.running_var[h:])
        return m


class GeMPool1d(nn.Module):
    """Paper Eq. (2) Generalized Mean Pooling — reduces one given axis with a learnable p."""

    def __init__(self, dim: int, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=self.dim).pow(1.0 / self.p)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, p={float(self.p.detach()):.3f}"


class LogSpectrogram(nn.Module):
    """8 kHz waveform → log spectrogram (linear) or log-mel. Paper §3.2: n_fft 512, hop 256."""

    def __init__(self, sample_rate: int = 8000, n_fft: int = 512, hop: int = 256,
                 use_mel: bool = False, n_mels: int = 128, eps: float = 1e-10):
        super().__init__()
        self.use_mel = bool(use_mel)
        self.eps = eps
        import torchaudio

        if self.use_mel:
            self.tf = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop,
                n_mels=n_mels, power=2.0, center=True, pad_mode="reflect",
            )
        else:
            self.tf = torchaudio.transforms.Spectrogram(
                n_fft=n_fft, win_length=n_fft, hop_length=hop, power=2.0, center=True, pad_mode="reflect",
            )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, 1, F, T)."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        s = self.tf(wav)
        return torch.log(s.clamp(min=self.eps)).unsqueeze(1)


@ModelFactory.register_model_class("resnest_ibn_dt")
class ResNeStIBNDomainTransfer(BaseRSCModel, nn.Module):
    """
    Architecture of the Domain Transfer paper §2.2. `baseline=True` reduces it to the Baseline of paper Table 2
    (backbone + average pooling + a single linear classifier).
    """

    def __init__(
        self,
        n_cls: int = 4,
        backbone: str = "resnest50d",       # assumption: depth not specified in the paper
        pretrained: bool = True,            # ImageNet pretraining
        ibn: bool = True,                   # assumption: IBN-a, stages 1–3
        ibn_stages: tuple = (1, 2, 3),
        baseline: bool = False,             # True → Baseline configuration of Table 2
        dropout: float = 0.5,               # paper §3.2 "All dropout rates are set to 0.5"
        gem_p: float = 3.0,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop: int = 256,
        use_mel: bool = False,
        n_mels: int = 128,
        mixup_alpha: float = 0.2,       # paper §2.1 Eq. (1)
        label_smoothing: float = 0.01,  # paper §3.2 "label smoothing 0.01"
    ):
        super().__init__()
        import timm

        self.baseline = bool(baseline)
        self.n_cls = int(n_cls)
        self.mixup_alpha = float(mixup_alpha)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("_class_weight", torch.ones(n_cls), persistent=False)
        self.spec = LogSpectrogram(sample_rate, n_fft, hop, use_mel, n_mels)

        net = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        self.feat_dim = net.num_features
        if self.feat_dim != 2048:
            logger.warning("The paper uses OutDim 2048 but %s has %d — the head input dimension changes", backbone, self.feat_dim)

        self._fold_stem_to_one_channel(net)
        if ibn:
            self._apply_ibn(net, ibn_stages)
        self.backbone = net

        if self.baseline:
            self.pool_f = None
            self.pool_t = None
            self.head = nn.Linear(self.feat_dim, n_cls)
        else:
            # paper: GeMPool(frequency axis) → GeMPool(time axis).  backbone output (B, C, F', T')
            self.pool_f = GeMPool1d(dim=2, p=gem_p)
            self.pool_t = GeMPool1d(dim=2, p=gem_p)  # once the frequency axis is removed, time is dim 2
            self.head = nn.Sequential(
                nn.BatchNorm1d(self.feat_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feat_dim, 512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, n_cls),
            )

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _fold_stem_to_one_channel(net: nn.Module) -> None:
        """Folds the first conv to 1-channel input (RGB weights summed over channels — AST/DASS convention)."""
        stem = getattr(net, "conv1", None)
        if isinstance(stem, nn.Sequential):
            first = stem[0]
        elif isinstance(stem, nn.Conv2d):
            first = stem
        else:
            raise RuntimeError("First conv not found — check the backbone structure")
        if first.in_channels == 1:
            return
        new = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size, stride=first.stride,
                        padding=first.padding, bias=first.bias is not None)
        with torch.no_grad():
            new.weight.copy_(first.weight.sum(dim=1, keepdim=True))
            if first.bias is not None:
                new.bias.copy_(first.bias)
        if isinstance(stem, nn.Sequential):
            stem[0] = new
        else:
            net.conv1 = new

    @staticmethod
    def _apply_ibn(net: nn.Module, stages) -> None:
        """Replaces bn1 of each block in the given stages with IBN-a (same position as IBN-Net resnet_ibn.py:71-74)."""
        n = 0
        for s in stages:
            layer = getattr(net, f"layer{s}", None)
            if layer is None:
                continue
            for blk in layer:
                bn1 = getattr(blk, "bn1", None)
                if isinstance(bn1, nn.BatchNorm2d):
                    blk.bn1 = IBN.from_batchnorm(bn1)
                    n += 1
        logger.info("Applied IBN-a: %d blocks (stages %s)", n, list(stages))

    # ------------------------------------------------------------------ forward
    def set_class_weight(self, w) -> None:
        """Injects the balanced class weight from the datamodule — paper §2.1 'higher weight for minority classes'."""
        self._class_weight = torch.as_tensor(w, dtype=torch.float, device=self._class_weight.device)

    def forward_from_spec(self, x: torch.Tensor) -> torch.Tensor:
        """Spectrogram → logits. Kept separate because mixup happens at the spectrogram stage."""
        x = self.backbone(x)                  # (B, C, F', T')
        if self.baseline:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        else:
            x = self.pool_f(x)                # reduce frequency axis → (B, C, T')
            x = self.pool_t(x)                # reduce time axis      → (B, C)
        return self.head(x)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward_from_spec(self.spec(wav))

    def _soft_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Applies label smoothing ε to one-hot — (1-ε)·onehot + ε/K (paper §3.2, ε=0.01)."""
        y = F.one_hot(labels, self.n_cls).to(torch.float)
        e = self.label_smoothing
        return y * (1.0 - e) + e / self.n_cls if e > 0 else y

    def _weighted_soft_ce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Soft-target CE + class weights.

        Uses the same definition as `F.cross_entropy(..., weight=w)` for hard labels —
        the per-sample loss −Σ_c y_c log p_c is multiplied by the sample weight Σ_c y_c w_c and **divided by the sum of weights**
        (same as PyTorch's weighted mean reduction). When mixup spreads y over two classes,
        the weights are mixed in the same proportion.
        """
        w = self._class_weight.to(logits.device)
        logp = F.log_softmax(logits, dim=-1)
        per_sample = -(y * logp).sum(dim=-1)
        sw = (y * w.unsqueeze(0)).sum(dim=-1)
        return (per_sample * sw).sum() / torch.clamp(sw.sum(), min=1e-12)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """
        Training path of paper §2.1·§3.2 — spectrogram mixup (α=0.2) + label smoothing 0.01 + class-weighted CE.

        mixup is applied **only during training**. In the re-fine-tuning phase of two-stage training a callback
        sets `mixup_alpha = 0` so only the original data and labels are used.
        """
        spec = self.spec(audio_batch)
        y = self._soft_targets(labels)
        if self.training and self.mixup_alpha > 0:
            spec, y = mixup_batch(spec, y, self.mixup_alpha)
        logits = self.forward_from_spec(spec)
        return logits, self._weighted_soft_ce(logits, y)
