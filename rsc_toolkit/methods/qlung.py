"""
QLung (Interspeech 2026, arXiv 2606.11915) method: a faithful port of the official code (RSC-Toolkit/QLung 46408dd) (official-code parity).

Components (line numbers refer to the official code at that commit):
- AngularClassifier (method/loss.py:98-115): L2-normalized features and weights → cosine × s_a (default 37, train_qlung.py:271).
- AngularMarginLoss = DFAM (method/loss.py:10-91): ArcFace-style margin with a separate weight W_d (n_cls×768, xavier_normal, gain 1).
    per-sample margin m_d = γ·(κ·AQS) + (1−γ)·m_c,  m_c = −log π_y · m_target / log 4   (loss.py:27-38, C=4 hardcoded → log_base_c)
    easy_margin=False rule: phi = cos−th>0 ? cos(θ+m_d) : cos−mm,  th = cos(π−m_d), mm = sin(π−m_d)·m_d (per sample; loss.py:71-87)
    output = one_hot·phi + (1−one_hot)·cos, × s_d (default 15, train_qlung.py:328).
    dtype quirk: one_hot.double() @ class_margin(float64) makes the margin float64 → cos/sin/phi/output are promoted to float64
    (even with an fp16 cosine under autocast, float64 is excluded from autocast). The same operation order is kept here.
- Total loss (train_qlung.py:463-470): CE(AngularClassifier logits .to(float64)) + λ·CE(DFAM logits), λ = --alpha 0.4.
- AQS (util/get_acoustic_features.py:16-27,52-78; util/icbhi_dataset.py:174-177): from the 8 s padded cycle waveform,
    rms = mean(librosa.feature.rms(1024/512)),  H = mean(normalized spectral entropy; STFT 1024/512, base 2, /log2(513))
    code formula: clip((−α·H_norm + β·R_norm + α)/(α+β), 0, 1),  R_norm = clip((rms − 0.001)/0.799, 0, 1), α 0.7, β 0.3.
    **Code bug (kept as the default)**: icbhi_dataset.py:177 passes (rms_energy, rms_energy), so rms takes the place of H
    → entropy_source="rms" (default, golden) | "entropy" (code formula + real entropy, matches the paper's Figure 1 distribution).
    formula="code" (default) | "paper" (paper Eq. (5) verbatim: clip(1 − α·H + β·R_norm, 0, 1); inconsistent with Figure 1, for reference).
    librosa is used only for the golden (tests/golden/qlung/aqs_golden.pt); the runtime re-implements it in torch (following librosa ≥0.10
    center=True, pad_mode="constant", periodic hann; golden comparison in tests/unit/models/test_ast_qlung.py).

Where the paper and the code disagree, the code is the default and the paper version is behind a flag.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ICBHI official train 4-class counts (train_qlung.py:278 comment / trainer/losses.py), for DFAM π_y
ICBHI_TRAIN_CLASS_NUMS = (2063, 1215, 501, 363)


# ----------------------------------------------------------------------------- AQS (acoustic quality score)
def rms_energy(y: torch.Tensor, frame_length: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """
    Frame mean of librosa.feature.rms(y, frame_length, hop_length): center=True, frame_length//2 zero padding on both ends,
    1 + len//hop frames. y: (T,) or (1,T) float. Returns a scalar tensor (keeps y's dtype).
    """
    y = y.reshape(-1)
    pad = frame_length // 2
    yp = F.pad(y, (pad, pad))
    frames = yp.unfold(0, frame_length, hop_length)          # (n_frames, frame_length)
    rms = torch.sqrt((frames * frames).mean(dim=1))
    return rms.mean()


def spectral_entropy(y: torch.Tensor, n_fft: int = 1024, hop_length: int = 512, pad_mode: str = "constant",
                     nan_policy: str = "propagate") -> torch.Tensor:
    """
    util/get_acoustic_features.py compute_spectral_entropy:
      S = |STFT(y)| (librosa defaults: hann(periodic), center=True, pad_mode="constant"(≥0.10)),
      ps = S / Σ_f S, H_frame = scipy.stats.entropy(ps, base=2) / log2(n_bins), mean over frames.
    scipy entropy renormalizes ps and treats 0·log0 = 0. Frames with Σ_f S = 0 (digital silence) become nan as in the original code, so
    the whole frame mean is nan (nan_policy="propagate"; 71 of the 6,898 ICBHI cycles). nan_policy="nanmean" averages finite frames only
    (toolkit extension; not in the official code).
    """
    y = y.reshape(-1)
    window = torch.hann_window(n_fft, periodic=True, dtype=y.dtype, device=y.device)
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window,
                      center=True, pad_mode=pad_mode, return_complex=True)
    S = spec.abs()                                            # (n_bins, n_frames)
    ps = S / S.sum(dim=0, keepdim=True)
    ps = ps / ps.sum(dim=0, keepdim=True)                     # scipy's internal renormalization (same values)
    ent = -(torch.special.xlogy(ps, ps)).sum(dim=0) / math.log(2.0)
    max_entropy = math.log2(S.shape[0])
    ent = ent / max_entropy
    if nan_policy == "nanmean":
        finite = ent[torch.isfinite(ent)]
        return finite.mean() if finite.numel() else ent.mean()
    if nan_policy != "propagate":
        raise ValueError(f"nan_policy must be propagate|nanmean, got {nan_policy!r}")
    return ent.mean()


def quality_score(entropy_norm, rms, rms_min: float = 0.001, rms_max: float = 0.8,
                  alpha: float = 0.7, beta: float = 0.3, formula: str = "code"):
    """
    formula="code": util/get_acoustic_features.py:52-78 compute_quality_score_entropy_rms —
        R = clip((rms−rms_min)/(rms_max−rms_min), 0, 1); raw = −α·H + β·R; clip((raw + α)/(α + β), 0, 1)
    formula="paper": paper Eq. (5) verbatim, clip(1 − α·H + β·R, 0, 1) (with α+β=1 it differs from the code formula by a +α·… constant)
    Scalar (float/tensor) input. Returns a float tensor.
    """
    e = torch.as_tensor(entropy_norm, dtype=torch.float64)
    r = torch.as_tensor(rms, dtype=torch.float64)
    rms_score = torch.clamp((r - rms_min) / (rms_max - rms_min), 0.0, 1.0)
    if formula == "code":
        raw = -alpha * e + beta * rms_score
        score_min, score_max = -alpha, beta
        return torch.clamp((raw - score_min) / (score_max - score_min), 0.0, 1.0)
    if formula == "paper":
        return torch.clamp(1.0 - alpha * e + beta * rms_score, 0.0, 1.0)
    raise ValueError(f"formula must be code|paper, got {formula!r}")


def compute_acoustic_features(wave: torch.Tensor) -> Dict[str, float]:
    """8 s padded cycle waveform (1,T)/(T,) → {'rms', 'entropy'} (icbhi_dataset.py:175-176)."""
    w = wave.detach().reshape(-1).float().cpu()
    return {"rms": float(rms_energy(w)), "entropy": float(spectral_entropy(w)),
            "entropy_nanmean": float(spectral_entropy(w, nan_policy="nanmean"))}


AQS_ENTROPY_SOURCES = ("rms", "entropy", "entropy_nanmean")


def aqs_from_features(rms: float, entropy: float, *, alpha: float = 0.7, beta: float = 0.3,
                      entropy_source: str = "rms", formula: str = "code", entropy_nanmean: Optional[float] = None) -> float:
    """
    icbhi_dataset.py:177: the default entropy_source="rms" keeps the compute_quality_score_entropy_rms(rms, rms, …) bug.
    entropy_source="entropy" uses (spectral_entropy, rms) literally (cycles with silent frames give nan → nan loss → AMP GradScaler skips the step and shrinks the scale),
    "entropy_nanmean" uses the mean entropy over finite frames (toolkit extension, no nan).
    """
    if entropy_source == "rms":
        h = rms
    elif entropy_source == "entropy":
        h = entropy
    elif entropy_source == "entropy_nanmean":
        if entropy_nanmean is None:
            raise ValueError("entropy_source=entropy_nanmean requires an entropy_nanmean value")
        h = entropy_nanmean
    else:
        raise ValueError(f"entropy_source must be one of {AQS_ENTROPY_SOURCES}, got {entropy_source!r}")
    return float(quality_score(h, rms, alpha=alpha, beta=beta, formula=formula))


def aqs_variant_name(alpha: float, beta: float, entropy_source: str, formula: str) -> str:
    return f"{entropy_source}_{formula}_a{alpha:g}_b{beta:g}"


# ----------------------------------------------------------------------------- angular classifier / DFAM
class AngularClassifier(nn.Module):
    """method/loss.py:98-115 — F.linear(F.normalize(x), F.normalize(W)) · s. W: xavier_normal(gain 1)."""

    def __init__(self, hidden_dim: int, n_cls: int, s: float = 15.0):
        super().__init__()
        self.n_cls = n_cls
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.empty(n_cls, hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine * self.s


def class_imbalance_margins(class_nums: Sequence[float], m_target: float = 0.2, log_base_c: float = 4.0,
                            eps: float = 1e-12) -> torch.Tensor:
    """
    method/loss.py:27-38: π_y = n_y/N, c = m_target / log(4) (C=4 hardcoded → log_base_c), m_c = −log π_y · c. float64.
    ICBHI defaults → about [0.1006, 0.1770, 0.3048, 0.3512].
    """
    counts = torch.tensor(list(class_nums), dtype=torch.float64)
    total = counts.sum().clamp_min(eps)
    pi = (counts / total).clamp_min(eps)
    c = float(m_target / torch.log(torch.tensor(log_base_c, dtype=torch.float64)))
    return -torch.log(pi) * c


class AngularMarginLoss(nn.Module):
    """
    DFAM (method/loss.py:10-91): same name as the original, but **returns logits** (CE is applied by the caller, train_qlung.py:467).
    The original hardcoded .cuda() is made device-agnostic; the dtype promotion order is unchanged.
    """

    def __init__(self, hidden_dim: int, n_cls: int, s: float = 15.0, m: float = 0.3, gamma: float = 0.5,
                 m_target: float = 0.2, class_nums: Optional[Sequence[float]] = None, easy_margin: bool = False,
                 log_base_c: float = 4.0):
        super().__init__()
        if class_nums is None:
            class_nums = ICBHI_TRAIN_CLASS_NUMS
        if len(class_nums) != n_cls:
            raise ValueError(f"class_nums length {len(class_nums)} != n_cls {n_cls}")
        self.m = m                      # κ (quality margin coefficient, original m=0.5)
        self.s = s
        self.n_cls = n_cls
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.m_target = m_target
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(n_cls, hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        # The original code holds a float64 tensor via .cuda() (outside state_dict). Here it is a non-persistent buffer, harmless to MA and checkpoints (constant).
        self.register_buffer("class_margin", class_imbalance_margins(class_nums, m_target, log_base_c), persistent=False)

    def margins(self, labels: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        """Per-sample m_d (float64): γ·(m·AQS) + (1−γ)·m_c[y]. Same operation order as loss.py:50-61."""
        one_hot = torch.zeros(labels.shape[0], self.n_cls, dtype=torch.float64, device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        class_margin = one_hot @ self.class_margin.to(labels.device)
        q = self.m * quality_score.to(labels.device)
        return self.gamma * q + (1 - self.gamma) * class_margin

    def forward(self, x: torch.Tensor, labels: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0 and torch.max(labels) < self.n_cls
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        one_hot = one_hot.double()

        class_margin = one_hot @ self.class_margin.to(one_hot.device)
        q = self.m * quality_score.to(cosine.device)
        margin = self.gamma * q + (1 - self.gamma) * class_margin

        cos_quality_m = torch.cos(margin).unsqueeze(1)
        sin_quality_m = torch.sin(margin).unsqueeze(1)
        th = torch.cos(torch.pi - margin).unsqueeze(1)
        mm = (torch.sin(torch.pi - margin) * margin).unsqueeze(1)

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * cos_quality_m - sine * sin_quality_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - th) > 0, phi, cosine - mm)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


def qlung_dual_loss(features: torch.Tensor, labels: torch.Tensor, quality_score: torch.Tensor,
                    classifier: AngularClassifier, dfam: AngularMarginLoss, loss_func, lambda_aqm: float = 0.4):
    """
    train_qlung.py:462-470 (method aqm):
      features_norm = F.normalize(features); output = classifier(features_norm)
      class_loss = criterion[0](output.to(float64), labels)          # criterion[0] = CE (unweighted)
      output2 = DFAM(features_norm, labels, quality_score); aqm_loss = CE(output2, labels) · λ
    Returns (output, class_loss + aqm_loss, class_loss, aqm_loss).
    """
    features_norm = F.normalize(features)
    output = classifier(features_norm)
    class_loss = loss_func(output.to(torch.float64), labels)
    output2 = dfam(features_norm, labels, quality_score)
    aqm_loss = F.cross_entropy(output2, labels) * lambda_aqm
    return output, class_loss + aqm_loss, class_loss, aqm_loss
