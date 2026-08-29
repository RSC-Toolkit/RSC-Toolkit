"""
Prototype Learning for Interpretable Respiratory Sound Analysis (Ren·Nguyen·Nejdl, ICASSP 2022) — 공식 코드 충실 이식 (P0).

출처:
- 논문: https://arxiv.org/abs/2110.03536 (ICASSP 2022, pp.9087–9091)
- 공식 코드: https://github.com/L3S/PrototypeSound (MIT, HEAD 5116de7)
  baseline_cnn/pytorch/models_pytorch.py (DecisionLevelMaxPooling / ProtoNet / CNN_encoder / PrototypeLayer),
  baseline_cnn/utils/config.py (sr 4000, cycle_len 16000, win 256, hop 128, mel 128)

구조(공식과 동일):
  파형 (B, 16000) @4 kHz
  → torchlibrosa Spectrogram(n_fft 256, hop 128, win 256, hann, center, reflect)
  → LogmelFilterBank(sr 4000, 128 mel, fmin 20, fmax 2000, ref 1.0, amin 1e-10, top_db None)   → (B,1,126,128)
  → Δ, ΔΔ 를 채널로 이어붙임(torch.diff, append=마지막 프레임)                                    → (B,3,126,128)
  → CNN_encoder: BatchNorm2d(128)(멜 축) + ConvBlock 3→64→128→256→512 (각 max-pool 2×2, dropout 0.2) → (B,512,7,8)
  → dropout 0.5
  → [ProtoNet] PrototypeLayer(cosine, proto_form) → similarity (B, n_proto) → Linear(n_proto, 4, bias=False) → log_softmax
    [CNN-8 기준선] max-pool 전체 → (B,512) → Linear(512, 4) → log_softmax

**torchlibrosa 를 그대로 쓴다**(.venv 에 설치, MIT, 무의존). PANNs 계열(cnn14_gapaug)의 torchaudio 재현본과는
max|diff| 8.9e-3 dB 차이가 있어, 이 논문은 공식과 비트 단위로 맞추기 위해 원 라이브러리를 사용한다.

원본 그대로 유지하는 quirk:
- **`np.float` 사용(공식 코드는 현대 numpy 에서 실행 불가)**: PrototypeLayer 가 `np.zeros(..., dtype=np.float)` 로
  버퍼를 만든 뒤 GPU 로 옮긴다. numpy≥1.24 에서 `np.float` 는 제거돼 AttributeError 가 난다. 여기서는 동일 결과를
  주는 `torch.zeros` 로 만들되 **연산 순서·수식은 그대로** 둔다(골든으로 고정).
- **배치·프로토타입 이중 루프**: 공식은 파이썬 루프로 코사인 유사도를 계산한다. 여기서는 수치가 동일함을
  골든으로 확인한 뒤 벡터화한다(루프판은 `_loop=True` 로 남겨 대조 가능).
- **`sim_lowbound` 하한 1e-8**: 코사인 분모를 `max(‖a‖‖b‖, 1e-8)` 로 클램프 — F.cosine_similarity(eps 1e-8 를
  각 노름에 더함)와 다른 동작이라 그대로 유지.
- **vector2d_maxp 는 BN 이 주석 처리**되어 있다(공식 코드 그대로). 논문 Table 3 의 "w/ BN" 행은 vector2d(EleSim)
  기준이며, MaxSim w/ BN 행은 Sp 0.00 으로 붕괴한 결과를 논문도 그대로 싣고 있다.
- diverse_loss 는 프로토타입이 1개/클래스일 때 internal 항이 0 이 되도록 분기한다(공식 `proto_per_class == 1`).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlibrosa as tl

from .base import BaseRSCModel
from .backbone.cnn import ConvBlock, _init_bn, _init_layer
from .factory import ModelFactory

logger = logging.getLogger(__name__)

PROTO_FORMS = ("vector1d", "vector2d", "vector2d_att", "vector2d_avgp", "vector2d_maxp", "vector2d_maxp_att")


class ProtoLogMel(nn.Module):
    """공식 config 그대로의 torchlibrosa 특징 추출 + Δ/ΔΔ 채널 결합."""

    def __init__(self, sample_rate: int = 4000, win_length: int = 256, hop_length: int = 128, mel_bins: int = 128,
                 fmin: int = 20, fmax: int = 2000):
        super().__init__()
        self.spectrogram_extractor = tl.Spectrogram(
            n_fft=win_length, hop_length=hop_length, win_length=win_length,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = tl.LogmelFilterBank(
            sr=sample_rate, n_fft=win_length, n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, 3, T, mel) — 원본 forward 앞부분과 동일."""
        x = self.logmel_extractor(self.spectrogram_extractor(wav))  # (B,1,T,M)
        b, c, _, m = x.shape
        x_diff1 = torch.diff(x, n=1, dim=2, append=x[:, :, -1, :].view((b, c, 1, m)))
        x_diff2 = torch.diff(x_diff1, n=1, dim=2, append=x_diff1[:, :, -1, :].view((b, c, 1, m)))
        return torch.cat((x, x_diff1, x_diff2), dim=1)


class CNNEncoder(nn.Module):
    """원본 CNN_encoder — bn0(멜 축) + ConvBlock ×4, 각 max-pool 2×2 + dropout 0.2."""

    def __init__(self, mel_bins: int = 128):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        _init_bn(self.bn0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for blk in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = blk(x, pool_size=(2, 2), pool_type="max")
            x = F.dropout(x, p=0.2, training=self.training)
        return x


def _cos(a: torch.Tensor, b: torch.Tensor, dim: int, lowbound: float = 1e-8) -> torch.Tensor:
    """원본의 코사인: sum(a*b)/max(‖a‖·‖b‖, 1e-8) — F.cosine_similarity 와 클램프 방식이 다르다."""
    num = torch.sum(a * b, dim=dim)
    den = torch.norm(a, p=2, dim=dim) * torch.norm(b, p=2, dim=dim)
    return num / torch.clamp(den, min=lowbound)


class PrototypeLayer(nn.Module):
    """원본 PrototypeLayer(distance='cosine') — 6개 proto_form 전부 이식."""

    def __init__(self, n_proto: int, proto_form: str, emb_ch: int = 512, emb_t: int = 7, emb_f: int = 8):
        super().__init__()
        if proto_form not in PROTO_FORMS:
            raise ValueError(f"proto_form must be one of {PROTO_FORMS}, got {proto_form!r}")
        self.n_proto = n_proto
        self.proto_form = proto_form
        self.emb_ch, self.emb_t, self.emb_f = emb_ch, emb_t, emb_f

        if proto_form == "vector1d":
            self.prototype = nn.Parameter(torch.empty((n_proto, emb_ch), dtype=torch.float), requires_grad=True)
        elif "att" in proto_form:
            self.prototype = nn.Parameter(torch.empty((n_proto, emb_ch, emb_t * emb_f), dtype=torch.float), requires_grad=True)
        else:
            self.prototype = nn.Parameter(torch.empty((n_proto, emb_ch, emb_t, emb_f), dtype=torch.float), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype.shape), requires_grad=False)  # 원본에 있으나 미사용
        if "att" in proto_form:
            self.fc = nn.Linear(emb_ch, 1)
        self.bn = nn.BatchNorm2d(n_proto)
        self.bn1d = nn.BatchNorm1d(n_proto)
        self.ln = nn.LayerNorm(n_proto)

        nn.init.xavier_uniform_(self.prototype)
        _init_bn(self.bn)
        _init_bn(self.bn1d)
        if "att" in proto_form:
            _init_layer(self.fc)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.proto_form
        p = self.prototype

        if f == "vector1d":
            # (B,512,T,F) → max-pool 전체 → (B,512,1,1) → flatten
            v = F.max_pool2d(x, kernel_size=x.shape[2:]).flatten(start_dim=1)      # (B,512)
            sim = _cos(v.unsqueeze(1), p.flatten(start_dim=1).unsqueeze(0), dim=2)  # (B,n_proto)
            return self.ln(sim), p

        if f == "vector2d":
            sim = _cos(x.unsqueeze(1), p.unsqueeze(0), dim=2)                       # (B,n_proto,T,F)
            sim = self.bn(sim)
            sim = F.avg_pool2d(sim, kernel_size=sim.shape[2:]).view(sim.shape[0:2])
            return sim, p

        if f == "vector2d_att":
            xf = torch.flatten(x, start_dim=2)                                      # (B,512,56)
            att = _cos(xf.unsqueeze(1), p.unsqueeze(0), dim=2)                      # (B,n_proto,56)
            att = F.softmax(att, dim=-1)
            # sum over 56: att * x * prototype  → (B,n_proto,512)
            sim = torch.sum(att.unsqueeze(2) * xf.unsqueeze(1) * p.unsqueeze(0), dim=3)
            sim = F.relu_(self.fc(sim)).view(sim.shape[0:2])
            return sim, p

        if f in ("vector2d_avgp", "vector2d_maxp"):
            # 모든 (t,f) × (t',f') 쌍의 코사인 → 평균/최대로 축약 (원본 dim 5 → dim 4 순서 유지)
            a = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)                          # (B,1,512,T,F,1,1)
            b = p.unsqueeze(0).unsqueeze(3).unsqueeze(4)                            # (1,n,512,1,1,T,F)
            sim_all = _cos(a, b, dim=2)                                             # (B,n,T,F,T,F)
            if f == "vector2d_avgp":
                sim = torch.mean(torch.mean(sim_all, 5), 4)
                sim = self.bn(sim)
            else:
                sim = torch.max(torch.max(sim_all, 5)[0], 4)[0]                     # 원본은 BN 주석 처리
            sim = F.avg_pool2d(sim, kernel_size=sim.shape[2:]).view(sim.shape[0:2])
            return sim, p

        # vector2d_maxp_att
        xf = torch.flatten(x, start_dim=2)                                          # (B,512,56)
        a = xf.unsqueeze(1).unsqueeze(4)                                            # (B,1,512,56,1)
        b = p.unsqueeze(0).unsqueeze(3)                                             # (1,n,512,1,56)
        sim_all = _cos(a, b, dim=2)                                                 # (B,n,56,56)
        att, ind = torch.max(sim_all, 3)                                            # (B,n,56)
        att = F.softmax(att, dim=-1)
        # 원본: prototype[j][:, ind[i,j,:]] — 배치·프로토타입별로 뽑은 열
        p_sel = torch.gather(
            p.unsqueeze(0).expand(xf.shape[0], -1, -1, -1), 3,
            ind.unsqueeze(2).expand(-1, -1, self.emb_ch, -1),
        )                                                                           # (B,n,512,56)
        sim = torch.sum(att.unsqueeze(2) * xf.unsqueeze(1) * p_sel, dim=-1)         # (B,n,512)
        sim = F.relu_(self.fc(sim)).view(sim.shape[0:2])
        return sim, p


def diverse_loss(proto: torch.Tensor, classes_num: int) -> torch.Tensor:
    """원본 ProtoNet.diverse_loss — 프로토타입 간 코사인 유사도로 다양성 페널티."""
    pt_flat = proto if proto.dim() == 2 else torch.flatten(proto, start_dim=1)
    n = pt_flat.shape[0]
    proto_per_class = int(n / classes_num)
    sim = _cos(pt_flat.unsqueeze(1), pt_flat.unsqueeze(0), dim=2)                   # (n,n)
    internal = pt_flat.new_zeros(())
    for i in range(classes_num):
        s = slice(i * proto_per_class, (i + 1) * proto_per_class)
        internal = internal + torch.sum(sim[s, s])
    external = torch.sum(sim) - internal
    diag = torch.sum(sim * torch.eye(n, device=sim.device, dtype=sim.dtype))
    if proto_per_class == 1:
        # 원본: 클래스당 1개면 internal 항이 대각뿐이라 external 만 남긴다
        internal_mean = pt_flat.new_zeros(())
    else:
        internal = internal - diag
        internal_mean = internal / (classes_num * (proto_per_class * proto_per_class - proto_per_class))
    external_mean = external / (n * n - classes_num * proto_per_class * proto_per_class)
    return external_mean - internal_mean


@ModelFactory.register_model_class("protonet")
class ProtoNetModel(BaseRSCModel, nn.Module):
    """
    ICASSP 2022 Prototype 논문 모델. method:
      "cnn8"  = DecisionLevelMaxPooling 기준선 (논문 Table 3 첫 행)
      "proto" = ProtoNet (proto_form 으로 변형 선택; 논문 "Ours" = vector2d)
    학습 손실은 가중 NLL(+ proto 일 때 0.1·diverse) — training_forward 가 돌려준다.
    """

    def __init__(
        self,
        n_cls: int = 4,
        method: str = "proto",              # "cnn8" | "proto"
        proto_form: str = "vector2d",       # 논문 "Ours" = vector2d (EleSim-Van w/ BN)
        n_proto_per_class: int = 1,
        sample_rate: int = 4000,
        win_length: int = 256,
        hop_length: int = 128,
        mel_bins: int = 128,
        fmin: int = 20,
        fmax: int = 2000,
        cycle_len: int = 16000,
        diverse_weight: float = 0.1,        # 원본 main: loss + 0.1 * loss_diverse
        class_weight: Optional[list] = None,  # None 이면 datamodule 이 계산한 값을 주입받는다
    ):
        super().__init__()
        if method not in {"cnn8", "proto"}:
            raise ValueError(f"method must be cnn8|proto, got {method!r}")
        self.method = method
        self.n_cls = n_cls
        self.diverse_weight = float(diverse_weight)

        self.logmel = ProtoLogMel(sample_rate, win_length, hop_length, mel_bins, fmin, fmax)
        self.cnn_encoder = CNNEncoder(mel_bins)

        # 임베딩 격자 크기(공식 config 에서 (7, 8)) — 4번의 2×2 max-pool
        t = (cycle_len // hop_length + 1) // 16
        f = mel_bins // 16
        if method == "proto":
            self.prototypelayer = PrototypeLayer(n_proto_per_class * n_cls, proto_form, emb_ch=512, emb_t=t, emb_f=f)
            self.fc_final = nn.Linear(n_proto_per_class * n_cls, n_cls, bias=False)
        else:
            self.prototypelayer = None
            self.fc_final = nn.Linear(512, n_cls)
        _init_layer(self.fc_final)

        self.register_buffer("_class_weight", torch.tensor(class_weight, dtype=torch.float) if class_weight else torch.ones(n_cls))
        self._class_weight_set = class_weight is not None

    def set_class_weight(self, w) -> None:
        """datamodule 의 balanced class weight 주입(원본 generator.calculate_class_weight)."""
        self._class_weight = torch.as_tensor(w, dtype=torch.float, device=self._class_weight.device)
        self._class_weight_set = True

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) 또는 (B, 1, L) → log-softmax 로짓."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        x = self.logmel(wav)
        emb = self.cnn_encoder(x)
        if self.method == "cnn8":
            out = F.max_pool2d(emb, kernel_size=emb.shape[2:]).view(emb.shape[0:2])
            return F.log_softmax(self.fc_final(out), dim=-1)
        emb = F.dropout(emb, p=0.5, training=self.training)
        sim, _ = self.prototypelayer(emb)
        return F.log_softmax(self.fc_final(sim), dim=-1)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        """원본 main_pytorch{,_prototype}.py 학습 루프 — 가중 NLL (+ proto 는 0.1·diverse)."""
        logits = self.forward(audio_batch)
        w = self._class_weight.to(logits.device)
        loss = F.nll_loss(logits, labels, weight=w)
        if self.training and self.method == "proto":
            loss = loss + self.diverse_weight * diverse_loss(self.prototypelayer.prototype, self.n_cls)
        return logits, loss
