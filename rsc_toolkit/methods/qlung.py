"""
QLung(Interspeech 2026, arXiv 2606.11915) 방법 — 공식 코드(RSC-Toolkit/QLung 46408dd) 충실 이식 (P0).

구성 요소 (출처 줄번호는 공식 클론 outputs/reference/qlung 기준):
- AngularClassifier (method/loss.py:98-115): 특징·가중치 L2 정규화 → 코사인 × s_a (기본 37, train_qlung.py:271).
- AngularMarginLoss = DFAM (method/loss.py:10-91): 별도 가중치 W_d(n_cls×768, xavier_normal, gain 1) 로 ArcFace 식 마진.
    샘플별 마진 m_d = γ·(κ·AQS) + (1−γ)·m_c,  m_c = −log π_y · m_target / log 4   (loss.py:27-38, C=4 하드코딩 → log_base_c)
    easy_margin=False 규칙: phi = cos−th>0 ? cos(θ+m_d) : cos−mm,  th = cos(π−m_d), mm = sin(π−m_d)·m_d (샘플별; loss.py:71-87)
    출력 = one_hot·phi + (1−one_hot)·cos, × s_d (기본 15, train_qlung.py:328).
    dtype quirk: one_hot.double() @ class_margin(float64) 로 마진이 float64 → cos/sin/phi/출력이 float64 로 승격된다
    (autocast 아래 fp16 cosine 이어도 float64 는 autocast 제외). 여기서도 같은 연산 순서를 유지한다.
- 총손실 (train_qlung.py:463-470): CE(AngularClassifier 로짓 .to(float64)) + λ·CE(DFAM 로짓), λ = --alpha 0.4.
- AQS (util/get_acoustic_features.py:16-27,52-78; util/icbhi_dataset.py:174-177): 8 s 패딩 사이클 파형에서
    rms = mean(librosa.feature.rms(1024/512)),  H = mean(정규화 스펙트럼 엔트로피; STFT 1024/512, base 2, /log2(513))
    코드식: clip((−α·H_norm + β·R_norm + α)/(α+β), 0, 1),  R_norm = clip((rms − 0.001)/0.799, 0, 1), α 0.7, β 0.3.
    **코드 버그(기본값으로 유지)**: icbhi_dataset.py:177 이 (rms_energy, rms_energy) 를 넘겨 H 자리에 rms 가 들어간다
    → entropy_source="rms" (기본, P0 골든) | "entropy" (코드식 + 진짜 엔트로피, 논문 Figure 1 분포와 일치).
    formula="code" (기본) | "paper" (논문 식 (5) 그대로: clip(1 − α·H + β·R_norm, 0, 1) — Figure 1 과 불일치, 참고용).
    librosa 계산은 골든(tests/golden/qlung/aqs_golden.pt)에만 쓰고 런타임은 torch 로 재구현한다 (librosa ≥0.10 의
    center=True, pad_mode="constant", periodic hann 을 따른다; 골든 대조 tests/unit/models/test_ast_qlung.py).

논문·코드 불일치와 저자 질문은 docs/AUTHOR_QUESTIONS_2026-08-26.md QLung 절 참고. 여기서는 코드 기본값 + 플래그.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ICBHI 공식 train 4-class 개수 (train_qlung.py:278 주석 / trainer/losses.py) — DFAM π_y
ICBHI_TRAIN_CLASS_NUMS = (2063, 1215, 501, 363)


# ----------------------------------------------------------------------------- AQS (음향 품질 점수)
def rms_energy(y: torch.Tensor, frame_length: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """
    librosa.feature.rms(y, frame_length, hop_length) 의 프레임 평균 — center=True, 양끝 frame_length//2 zero 패딩,
    프레임 수 1 + len//hop. y: (T,) 또는 (1,T) float. 스칼라 텐서 반환 (y 의 dtype 유지).
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
      S = |STFT(y)| (librosa 기본: hann(periodic), center=True, pad_mode="constant"(≥0.10)),
      ps = S / Σ_f S, H_frame = scipy.stats.entropy(ps, base=2) / log2(n_bins), 프레임 평균.
    scipy entropy 는 ps 를 다시 정규화하고 0·log0 = 0 으로 처리한다. Σ_f S = 0 인 프레임(디지털 무음)은 원본처럼 nan 이 되어
    프레임 평균 전체가 nan 이다 (nan_policy="propagate"; ICBHI 6,898 사이클 중 71 건). nan_policy="nanmean" 은 유한 프레임만 평균한다
    (툴킷 확장 — 공식 코드에 없음).
    """
    y = y.reshape(-1)
    window = torch.hann_window(n_fft, periodic=True, dtype=y.dtype, device=y.device)
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window,
                      center=True, pad_mode=pad_mode, return_complex=True)
    S = spec.abs()                                            # (n_bins, n_frames)
    ps = S / S.sum(dim=0, keepdim=True)
    ps = ps / ps.sum(dim=0, keepdim=True)                     # scipy 내부 재정규화 (동일값)
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
    formula="paper": 논문 식 (5) 문자 그대로 clip(1 − α·H + β·R, 0, 1) (α+β=1 이면 코드식보다 +α·… 상수만큼 다름)
    스칼라(float/텐서) 입력. float 텐서로 반환.
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
    """8 s 패딩 사이클 파형 (1,T)/(T,) → {'rms', 'entropy'} (icbhi_dataset.py:175-176)."""
    w = wave.detach().reshape(-1).float().cpu()
    return {"rms": float(rms_energy(w)), "entropy": float(spectral_entropy(w)),
            "entropy_nanmean": float(spectral_entropy(w, nan_policy="nanmean"))}


AQS_ENTROPY_SOURCES = ("rms", "entropy", "entropy_nanmean")


def aqs_from_features(rms: float, entropy: float, *, alpha: float = 0.7, beta: float = 0.3,
                      entropy_source: str = "rms", formula: str = "code", entropy_nanmean: Optional[float] = None) -> float:
    """
    icbhi_dataset.py:177 — 기본 entropy_source="rms" 는 compute_quality_score_entropy_rms(rms, rms, …) 버그 그대로.
    entropy_source="entropy" 는 (spectral_entropy, rms) 문자 그대로(무음 프레임 사이클은 nan → nan 손실 → AMP GradScaler 가 스텝을 건너뜀·스케일 축소),
    "entropy_nanmean" 은 유한 프레임 평균 엔트로피(툴킷 확장, nan 없음).
    """
    if entropy_source == "rms":
        h = rms
    elif entropy_source == "entropy":
        h = entropy
    elif entropy_source == "entropy_nanmean":
        if entropy_nanmean is None:
            raise ValueError("entropy_source=entropy_nanmean 에는 entropy_nanmean 값이 필요하다")
        h = entropy_nanmean
    else:
        raise ValueError(f"entropy_source must be one of {AQS_ENTROPY_SOURCES}, got {entropy_source!r}")
    return float(quality_score(h, rms, alpha=alpha, beta=beta, formula=formula))


def aqs_variant_name(alpha: float, beta: float, entropy_source: str, formula: str) -> str:
    return f"{entropy_source}_{formula}_a{alpha:g}_b{beta:g}"


# ----------------------------------------------------------------------------- 각도 분류기 / DFAM
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
    method/loss.py:27-38: π_y = n_y/N, c = m_target / log(4) (C=4 하드코딩 → log_base_c), m_c = −log π_y · c. float64.
    ICBHI 기본값 → [0.1006, 0.1770, 0.3048, 0.3512] 근방.
    """
    counts = torch.tensor(list(class_nums), dtype=torch.float64)
    total = counts.sum().clamp_min(eps)
    pi = (counts / total).clamp_min(eps)
    c = float(m_target / torch.log(torch.tensor(log_base_c, dtype=torch.float64)))
    return -torch.log(pi) * c


class AngularMarginLoss(nn.Module):
    """
    DFAM (method/loss.py:10-91) — 이름은 원본 그대로이나 **로짓을 돌려준다** (CE 는 호출측, train_qlung.py:467).
    원본의 .cuda() 하드코딩은 device-agnostic 으로 바꾸되 dtype 승격 순서는 동일.
    """

    def __init__(self, hidden_dim: int, n_cls: int, s: float = 15.0, m: float = 0.3, gamma: float = 0.5,
                 m_target: float = 0.2, class_nums: Optional[Sequence[float]] = None, easy_margin: bool = False,
                 log_base_c: float = 4.0):
        super().__init__()
        if class_nums is None:
            class_nums = ICBHI_TRAIN_CLASS_NUMS
        if len(class_nums) != n_cls:
            raise ValueError(f"class_nums 길이 {len(class_nums)} != n_cls {n_cls}")
        self.m = m                      # κ (품질 마진 계수, 원본 m=0.5)
        self.s = s
        self.n_cls = n_cls
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.m_target = m_target
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(n_cls, hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        # 원본은 float64 텐서를 .cuda() 로 들고 있다(state_dict 밖). 여기서는 buffer 로 두되 MA·ckpt 에는 무해(상수).
        self.register_buffer("class_margin", class_imbalance_margins(class_nums, m_target, log_base_c), persistent=False)

    def margins(self, labels: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        """샘플별 m_d (float64): γ·(m·AQS) + (1−γ)·m_c[y]. loss.py:50-61 과 동일 연산 순서."""
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
      class_loss = criterion[0](output.to(float64), labels)          # criterion[0] = CE (비가중)
      output2 = DFAM(features_norm, labels, quality_score); aqm_loss = CE(output2, labels) · λ
    반환 (output, class_loss + aqm_loss, class_loss, aqm_loss).
    """
    features_norm = F.normalize(features)
    output = classifier(features_norm)
    class_loss = loss_func(output.to(torch.float64), labels)
    output2 = dfam(features_norm, labels, quality_score)
    aqm_loss = F.cross_entropy(output2, labels) * lambda_aqm
    return output, class_loss + aqm_loss, class_loss, aqm_loss
