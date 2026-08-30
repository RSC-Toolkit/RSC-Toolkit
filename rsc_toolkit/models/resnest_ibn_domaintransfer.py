"""
Domain Transfer (Wang & Wang, ICASSP 2022) 의 분류기 — **논문 명세 재구현** (공식·비공식 코드 없음, 등급 C).

출처: "A Domain Transfer Based Data Augmentation Method for Automated Respiratory Classification",
ICASSP 2022 pp.9017–9020, DOI 10.1109/ICASSP43922.2022.9746941 (§2.2 NetWork Structure).

논문이 명시한 구조(§2.2 마지막 문장 그대로):
  backbone(OutDim 2048) → GeMPool(주파수 축) → GeMPool(시간 축) → BatchNorm → Dropout
  → Linear(2048→512) → ReLU → Dropout → Linear(512→512) → ReLU → Dropout → Linear(512→4)
- 백본은 **ImageNet 사전학습 ResNeSt + IBN 모듈**. dropout 은 전부 0.5.
- GeMPool: F = (mean(x^p))^(1/p), p 는 학습 파라미터(논문 식 2). p=1 이면 average, p→∞ 면 max pooling.

가정(논문 미기재 — 전부 인자로 노출, AUTHOR_QUESTIONS 참조):
- **ResNeSt 변형**: 논문은 "ResNeSt" 라고만 하고 깊이를 밝히지 않는다. 출력 2048 은 50d/101e/200e/269e 모두 해당.
  기본값 `resnest50d`(가장 표준적·가벼움)로 두고 `backbone` 인자로 교체 가능.
- **IBN 종류·위치**: 논문은 "IBN 모듈" 이라고만 한다. IBN-Net 원논문의 기본 구성인 **IBN-a 를 stage 1–3 의 bn1 에 적용**
  (`ibn_cfg=('a','a','a',None)`, IBN-Net `resnet_ibn.py:115`)을 채택. **공개된 ResNeSt-IBN ImageNet 가중치는 존재하지
  않는다**(IBN-Net 은 ResNet/ResNeXt/SE-ResNet/DenseNet 만 배포, timm 은 IBN 없음) → timm ResNeSt ImageNet 가중치를
  로드한 뒤 bn1 을 IBN 으로 교체하고, 원 BN 의 affine 파라미터를 앞 절반=IN / 뒤 절반=BN 으로 나눠 승계한다
  (IN 은 running stats 를 쓰지 않으므로 그 절반의 통계는 버려진다).
- **입력 채널**: ImageNet 백본은 3채널인데 스펙트로그램은 1채널이다. AST·DASS 계열의 관행대로 첫 conv 가중치를
  채널 축으로 합산해 1채널로 접는다(`torch.sum(w, dim=1, keepdim=True)`).
- **스펙트로그램 종류**: 논문은 "STFT with 512 FFT, 256 hop" 이라고만 하고 mel 여부를 밝히지 않는다. Fig.1 캡션이
  "log-spectrogram" 이므로 **선형 STFT 크기의 로그**를 기본으로 하고, `use_mel=True` 로 log-mel 도 선택 가능.
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
    """IBN-Net(arXiv 1807.09441) `ibnnet/modules.py` 의 IBN 레이어 — 채널 절반 IN / 절반 BN."""

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
        """사전학습된 BatchNorm 의 affine 을 앞 절반=IN / 뒤 절반=BN 으로 승계(IN 은 running stats 없음)."""
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
    """논문 식 (2) Generalized Mean Pooling — 지정한 축 하나를 학습 가능한 p 로 축약."""

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
    """8 kHz 파형 → log 스펙트로그램(선형) 또는 log-mel. 논문 §3.2: n_fft 512, hop 256."""

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
    Domain Transfer 논문 §2.2 구조. `baseline=True` 면 논문 Table 2 의 Baseline
    (백본 + average pooling + 선형 분류기 하나)로 축약한다.
    """

    def __init__(
        self,
        n_cls: int = 4,
        backbone: str = "resnest50d",       # 가정: 논문은 깊이 미기재
        pretrained: bool = True,            # ImageNet 사전학습
        ibn: bool = True,                   # 가정: IBN-a, stage 1–3
        ibn_stages: tuple = (1, 2, 3),
        baseline: bool = False,             # True → Table 2 의 Baseline 구성
        dropout: float = 0.5,               # 논문 §3.2 "All dropout rates are set to 0.5"
        gem_p: float = 3.0,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop: int = 256,
        use_mel: bool = False,
        n_mels: int = 128,
        mixup_alpha: float = 0.2,       # 논문 §2.1 식 (1)
        label_smoothing: float = 0.01,  # 논문 §3.2 "label smoothing 0.01"
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
            logger.warning("논문은 OutDim 2048 인데 %s 는 %d — 헤드 입력 차원이 달라진다", backbone, self.feat_dim)

        self._fold_stem_to_one_channel(net)
        if ibn:
            self._apply_ibn(net, ibn_stages)
        self.backbone = net

        if self.baseline:
            self.pool_f = None
            self.pool_t = None
            self.head = nn.Linear(self.feat_dim, n_cls)
        else:
            # 논문: GeMPool(주파수 축) → GeMPool(시간 축).  백본 출력 (B, C, F', T')
            self.pool_f = GeMPool1d(dim=2, p=gem_p)
            self.pool_t = GeMPool1d(dim=2, p=gem_p)  # 주파수 축이 사라진 뒤에는 시간이 dim 2
            self.head = nn.Sequential(
                nn.BatchNorm1d(self.feat_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feat_dim, 512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, n_cls),
            )

    # ------------------------------------------------------------------ 구성
    @staticmethod
    def _fold_stem_to_one_channel(net: nn.Module) -> None:
        """첫 conv 를 1채널 입력으로 접는다(RGB 가중치 채널 합산 — AST·DASS 계열 관행)."""
        stem = getattr(net, "conv1", None)
        if isinstance(stem, nn.Sequential):
            first = stem[0]
        elif isinstance(stem, nn.Conv2d):
            first = stem
        else:
            raise RuntimeError("첫 conv 를 찾지 못했다 — 백본 구조 확인 필요")
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
        """지정 stage 의 각 블록 bn1 을 IBN-a 로 교체(IBN-Net resnet_ibn.py:71-74 와 같은 위치)."""
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
        logger.info("IBN-a 적용: %d 블록 (stage %s)", n, list(stages))

    # ------------------------------------------------------------------ forward
    def set_class_weight(self, w) -> None:
        """datamodule 의 balanced class weight 주입 — 논문 §2.1 '소수 클래스 가중치 ↑'."""
        self._class_weight = torch.as_tensor(w, dtype=torch.float, device=self._class_weight.device)

    def forward_from_spec(self, x: torch.Tensor) -> torch.Tensor:
        """스펙트로그램 → 로짓. mixup 이 스펙트로그램 단계에서 일어나므로 분리해 둔다."""
        x = self.backbone(x)                  # (B, C, F', T')
        if self.baseline:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        else:
            x = self.pool_f(x)                # 주파수 축 축약 → (B, C, T')
            x = self.pool_t(x)                # 시간 축 축약   → (B, C)
        return self.head(x)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward_from_spec(self.spec(wav))

    def _soft_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """one-hot 에 label smoothing ε 적용 — (1-ε)·onehot + ε/K (논문 §3.2, ε=0.01)."""
        y = F.one_hot(labels, self.n_cls).to(torch.float)
        e = self.label_smoothing
        return y * (1.0 - e) + e / self.n_cls if e > 0 else y

    def _weighted_soft_ce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        soft target CE + 클래스 가중.

        하드 라벨일 때 `F.cross_entropy(..., weight=w)` 와 같은 정의를 쓴다 —
        샘플 손실 −Σ_c y_c log p_c 에 샘플 가중 Σ_c y_c w_c 를 곱하고 **가중 합으로 나눈다**
        (PyTorch 의 weighted mean reduction 과 동일). mixup 으로 y 가 두 클래스에 걸치면
        가중치도 같은 비율로 섞인다.
        """
        w = self._class_weight.to(logits.device)
        logp = F.log_softmax(logits, dim=-1)
        per_sample = -(y * logp).sum(dim=-1)
        sw = (y * w.unsqueeze(0)).sum(dim=-1)
        return (per_sample * sw).sum() / torch.clamp(sw.sum(), min=1e-12)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """
        논문 §2.1·§3.2 학습 경로 — 스펙트로그램 mixup(α=0.2) + label smoothing 0.01 + 클래스 가중 CE.

        mixup 은 **학습 때만** 적용한다. 2단계 학습의 재미세조정 구간에서는 콜백이
        `mixup_alpha = 0` 으로 꺼서 원본 데이터·원본 라벨만 쓰게 한다.
        """
        spec = self.spec(audio_batch)
        y = self._soft_targets(labels)
        if self.training and self.mixup_alpha > 0:
            spec, y = mixup_batch(spec, y, self.mixup_alpha)
        logits = self.forward_from_spec(spec)
        return logits, self._weighted_soft_ce(logits, y)
