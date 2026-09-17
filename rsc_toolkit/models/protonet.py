"""
Prototype Learning for Interpretable Respiratory Sound Analysis (Ren·Nguyen·Nejdl, ICASSP 2022) — exact port of the official code.

Sources:
- Paper: https://arxiv.org/abs/2110.03536 (ICASSP 2022, pp.9087–9091)
- Official code: https://github.com/L3S/PrototypeSound (MIT, HEAD 5116de7)
  baseline_cnn/pytorch/models_pytorch.py (DecisionLevelMaxPooling / ProtoNet / CNN_encoder / PrototypeLayer),
  baseline_cnn/utils/config.py (sr 4000, cycle_len 16000, win 256, hop 128, mel 128)

Architecture (same as official):
  waveform (B, 16000) @4 kHz
  → torchlibrosa Spectrogram(n_fft 256, hop 128, win 256, hann, center, reflect)
  → LogmelFilterBank(sr 4000, 128 mel, fmin 20, fmax 2000, ref 1.0, amin 1e-10, top_db None)   → (B,1,126,128)
  → Δ, ΔΔ concatenated as channels (torch.diff, append=last frame)                               → (B,3,126,128)
  → CNN_encoder: BatchNorm2d(128)(mel axis) + ConvBlock 3→64→128→256→512 (each max-pool 2×2, dropout 0.2) → (B,512,7,8)
  → dropout 0.5
  → [ProtoNet] PrototypeLayer(cosine, proto_form) → similarity (B, n_proto) → Linear(n_proto, 4, bias=False) → log_softmax
    [CNN-8 baseline] global max-pool → (B,512) → Linear(512, 4) → log_softmax

**torchlibrosa is used as is** (installed in .venv, MIT, no dependencies). The torchaudio reproduction used for the PANNs family (cnn14_gapaug)
differs by max|diff| 8.9e-3 dB, so this paper uses the original library to match the official code bit for bit.

Quirks kept from the original code:
- **`np.float` (the official code does not run on modern numpy)**: PrototypeLayer builds a buffer with `np.zeros(..., dtype=np.float)`
  and moves it to the GPU. `np.float` was removed in numpy≥1.24, which raises AttributeError. Here it is built with `torch.zeros`,
  which gives the same result, while **the order of operations and formulas are unchanged** (pinned by goldens).
- **Double loop over batch and prototypes**: the official code computes cosine similarity in Python loops. Here it is vectorized
  after confirming identical numbers against goldens.
- **`sim_lowbound` lower bound 1e-8**: the cosine denominator is clamped as `max(‖a‖‖b‖, 1e-8)` — this differs from F.cosine_similarity
  (which applies eps 1e-8 to each norm), so it is kept.
- **BN is commented out for vector2d_maxp** (as in the official code). The "w/ BN" row of paper Table 3 refers to vector2d (EleSim),
  and the paper itself reports the MaxSim w/ BN row collapsing to Sp 0.00.
- diverse_loss branches so that the internal term is 0 when there is one prototype per class (official `proto_per_class == 1`).
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
    """torchlibrosa feature extraction with the official config + Δ/ΔΔ channel concatenation."""

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
        """(B, L) → (B, 3, T, mel) — same as the first part of the original forward."""
        x = self.logmel_extractor(self.spectrogram_extractor(wav))  # (B,1,T,M)
        b, c, _, m = x.shape
        x_diff1 = torch.diff(x, n=1, dim=2, append=x[:, :, -1, :].view((b, c, 1, m)))
        x_diff2 = torch.diff(x_diff1, n=1, dim=2, append=x_diff1[:, :, -1, :].view((b, c, 1, m)))
        return torch.cat((x, x_diff1, x_diff2), dim=1)


class CNNEncoder(nn.Module):
    """Original CNN_encoder — bn0 (mel axis) + ConvBlock ×4, each max-pool 2×2 + dropout 0.2."""

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
    """Original cosine: sum(a*b)/max(‖a‖·‖b‖, 1e-8) — clamps differently from F.cosine_similarity."""
    num = torch.sum(a * b, dim=dim)
    den = torch.norm(a, p=2, dim=dim) * torch.norm(b, p=2, dim=dim)
    return num / torch.clamp(den, min=lowbound)


class PrototypeLayer(nn.Module):
    """Original PrototypeLayer(distance='cosine') — all 6 proto_forms ported."""

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

        self.ones = nn.Parameter(torch.ones(self.prototype.shape), requires_grad=False)  # present in the original but unused
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
            # (B,512,T,F) → global max-pool → (B,512,1,1) → flatten
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
            # cosine for every (t,f) × (t',f') pair → reduced by mean/max (original dim 5 → dim 4 order kept)
            a = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)                          # (B,1,512,T,F,1,1)
            b = p.unsqueeze(0).unsqueeze(3).unsqueeze(4)                            # (1,n,512,1,1,T,F)
            sim_all = _cos(a, b, dim=2)                                             # (B,n,T,F,T,F)
            if f == "vector2d_avgp":
                sim = torch.mean(torch.mean(sim_all, 5), 4)
                sim = self.bn(sim)
            else:
                sim = torch.max(torch.max(sim_all, 5)[0], 4)[0]                     # BN commented out in the original
            sim = F.avg_pool2d(sim, kernel_size=sim.shape[2:]).view(sim.shape[0:2])
            return sim, p

        # vector2d_maxp_att
        xf = torch.flatten(x, start_dim=2)                                          # (B,512,56)
        a = xf.unsqueeze(1).unsqueeze(4)                                            # (B,1,512,56,1)
        b = p.unsqueeze(0).unsqueeze(3)                                             # (1,n,512,1,56)
        sim_all = _cos(a, b, dim=2)                                                 # (B,n,56,56)
        att, ind = torch.max(sim_all, 3)                                            # (B,n,56)
        att = F.softmax(att, dim=-1)
        # original: prototype[j][:, ind[i,j,:]] — columns selected per batch item and prototype
        p_sel = torch.gather(
            p.unsqueeze(0).expand(xf.shape[0], -1, -1, -1), 3,
            ind.unsqueeze(2).expand(-1, -1, self.emb_ch, -1),
        )                                                                           # (B,n,512,56)
        sim = torch.sum(att.unsqueeze(2) * xf.unsqueeze(1) * p_sel, dim=-1)         # (B,n,512)
        sim = F.relu_(self.fc(sim)).view(sim.shape[0:2])
        return sim, p


def diverse_loss(proto: torch.Tensor, classes_num: int) -> torch.Tensor:
    """Original ProtoNet.diverse_loss — diversity penalty from cosine similarity between prototypes."""
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
        # original: with one prototype per class the internal term is only the diagonal, so only external remains
        internal_mean = pt_flat.new_zeros(())
    else:
        internal = internal - diag
        internal_mean = internal / (classes_num * (proto_per_class * proto_per_class - proto_per_class))
    external_mean = external / (n * n - classes_num * proto_per_class * proto_per_class)
    return external_mean - internal_mean


@ModelFactory.register_model_class("protonet")
class ProtoNetModel(BaseRSCModel, nn.Module):
    """
    Model of the ICASSP 2022 Prototype paper. method:
      "cnn8"  = DecisionLevelMaxPooling baseline (first row of paper Table 3)
      "proto" = ProtoNet (variant selected by proto_form; paper "Ours" = vector2d)
    The training loss is weighted NLL (+ 0.1·diverse for proto) — returned by training_forward.
    """

    def __init__(
        self,
        n_cls: int = 4,
        method: str = "proto",              # "cnn8" | "proto"
        proto_form: str = "vector2d",       # paper "Ours" = vector2d (EleSim-Van w/ BN)
        n_proto_per_class: int = 1,
        sample_rate: int = 4000,
        win_length: int = 256,
        hop_length: int = 128,
        mel_bins: int = 128,
        fmin: int = 20,
        fmax: int = 2000,
        cycle_len: int = 16000,
        diverse_weight: float = 0.1,        # original main: loss + 0.1 * loss_diverse
        class_weight: Optional[list] = None,  # if None, the value computed by the datamodule is injected
    ):
        super().__init__()
        if method not in {"cnn8", "proto"}:
            raise ValueError(f"method must be cnn8|proto, got {method!r}")
        self.method = method
        self.n_cls = n_cls
        self.diverse_weight = float(diverse_weight)

        self.logmel = ProtoLogMel(sample_rate, win_length, hop_length, mel_bins, fmin, fmax)
        self.cnn_encoder = CNNEncoder(mel_bins)

        # embedding grid size ((7, 8) with the official config) — four 2×2 max-pools
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
        """Injects the balanced class weight from the datamodule (original generator.calculate_class_weight)."""
        self._class_weight = torch.as_tensor(w, dtype=torch.float, device=self._class_weight.device)
        self._class_weight_set = True

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) or (B, 1, L) → log-softmax logits."""
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
        """Training loop of the original main_pytorch{,_prototype}.py — weighted NLL (+ 0.1·diverse for proto)."""
        logits = self.forward(audio_batch)
        w = self._class_weight.to(logits.device)
        loss = F.nll_loss(logits, labels, weight=w)
        if self.training and self.method == "proto":
            loss = loss + self.diverse_weight * diverse_loss(self.prototypelayer.prototype, self.n_cls)
        return logits, loss
