"""
Domain Transfer(Wang & Wang, ICASSP 2022) 의 오디오 증강 4종 — 논문 명세 재구현 (코드 없음, 등급 C).

출처: "A Domain Transfer Based Data Augmentation Method for Automated Respiratory Classification",
ICASSP 2022 pp.9017–9020, DOI 10.1109/ICASSP43922.2022.9746941 (§2.1).

논문이 명시한 것:
- 전통 증강: **잡음 추가(백색잡음 + 사전 녹음 환경잡음, SNR 을 1–10 에서 무작위 추출)**, time stretching, pitch shifting
- **Splicing**: 같은 클래스의 짧은 샘플들을 이어붙임 (클래스마다 같은 개수의 splice 샘플 생성 → 불균형 완화)
- **Circular padding**: zero padding 을 대체하며 **test set 에도 적용**
- **Mixup**: 스펙트로그램·라벨 선형결합, α = 0.2

가정(논문 미기재 — 전부 플래그로 노출, AUTHOR_QUESTIONS 참조):
- **SNR 단위는 dB**, 1–10 dB 를 균등 추출로 해석(논문은 "randomly sampled between 1 and 10" 만 기재).
- **환경잡음은 사용하지 않는다(백색잡음만)** — 논문이 어떤 코퍼스인지 밝히지 않았고, 툴킷 데이터 범위(ICBHI·SPRSound)
  밖의 외부 코퍼스를 끌어오지 않기 위함(사용자 결정 2026-08-30: "가장 단순한 형태로"). 표에 '환경잡음 미적용' 병기.
- time stretch / pitch shift 의 배율 범위 미기재 → 흔한 값(0.9–1.1 배속, ±2 semitone)을 기본값으로 두고 플래그화.
- splice 대상 "짧은 샘플" 기준·최대 개수 미기재 → 목표 길이 미만인 사이클을 목표 길이까지 이어붙이는 것으로 해석.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def add_white_noise(x: torch.Tensor, snr_db: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    목표 SNR(dB)이 되도록 백색 가우시안 잡음을 더한다.

    백색잡음 = 각 샘플이 서로 독립인 정규분포 난수(주파수축에서 전 대역 평탄한 전력 스펙트럼).
    스케일은 신호 전력에 맞춰 정한다:
        P_x = mean(x²)                     신호 전력
        P_n = P_x / 10^(SNR_dB / 10)       목표 잡음 전력
        n   = N(0,1) 난수 → P_n 이 되도록 재스케일
        out = x + n
    SNR 1 dB 는 잡음 전력이 신호의 약 79 %, 10 dB 는 10 % 다(논문의 "strong noise augmentation" 과 부합).
    무음 구간(P_x = 0)은 잡음만 남는 것을 막기 위해 원본을 그대로 돌려준다.
    """
    p_x = torch.mean(x.float() ** 2)
    if float(p_x) <= 0.0:
        return x
    p_n = p_x / (10.0 ** (snr_db / 10.0))
    n = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    n = n * torch.sqrt(p_n / torch.clamp(torch.mean(n.float() ** 2), min=1e-12))
    return x + n


def sample_snr_db(low: float = 1.0, high: float = 10.0, generator: Optional[torch.Generator] = None) -> float:
    """논문 §2.1 'SNR was randomly sampled between 1 and 10' — dB 균등 추출로 해석(가정)."""
    return float(torch.empty(1).uniform_(low, high, generator=generator).item())


def circular_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    논문 §2.1 Circular Padding — zero padding 대신 신호를 순환 반복해 목표 길이를 채운다.
    **test set 에도 적용**(논문 명시). 길면 앞에서 자른다(논문 'truncate the too-long audio').
    """
    n = int(x.shape[-1])
    if n == target_len:
        return x
    if n > target_len:
        return x[..., :target_len]
    reps = -(-target_len // n)  # ceil
    return x.repeat(*([1] * (x.dim() - 1)), reps)[..., :target_len]


def splice_same_class(clips: Sequence[torch.Tensor], target_len: int) -> torch.Tensor:
    """
    논문 §2.1 Splicing Audio — **같은 클래스**의 짧은 사이클들을 이어붙여 목표 길이를 만든다.
    모자라면 순환 패딩으로 마무리(논문이 splice 후 길이 처리를 따로 적지 않아 circular padding 과 일관되게 처리).
    """
    if not clips:
        raise ValueError("splice_same_class: 빈 clip 목록")
    out = torch.cat([c.reshape(1, -1) for c in clips], dim=-1)
    return circular_pad(out, target_len)


def mixup_batch(
    x: torch.Tensor, y_onehot: torch.Tensor, alpha: float = 0.2, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    논문 식 (1) — 스펙트로그램과 one-hot 라벨을 같은 λ 로 선형결합. λ ~ Beta(α, α), α = 0.2.
    배치 안에서 무작위 순열 짝을 쓴다(논문은 'randomly draw another sample xb from the dataset').
    """
    if alpha <= 0:
        return x, y_onehot
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.shape[0], device=x.device, generator=generator)
    return lam * x + (1.0 - lam) * x[idx], lam * y_onehot + (1.0 - lam) * y_onehot[idx]


def time_stretch(x: torch.Tensor, rate: float) -> torch.Tensor:
    """
    time stretching — 배율 rate 로 길이를 바꾼다(선형 보간 리샘플; 피치도 함께 변한다).
    논문은 배율 범위를 적지 않았다(가정: 0.9–1.1).
    """
    n = int(x.shape[-1])
    new_n = max(1, int(round(n / rate)))
    return torch.nn.functional.interpolate(
        x.reshape(1, 1, n).float(), size=new_n, mode="linear", align_corners=False
    ).reshape(1, new_n).to(x.dtype)


def pitch_shift(x: torch.Tensor, n_steps: float, sample_rate: int) -> torch.Tensor:
    """
    pitch shifting — 리샘플로 피치를 옮긴 뒤 원 길이로 되돌린다(길이 보존).
    논문은 반음 범위를 적지 않았다(가정: ±2 semitone).
    """
    n = int(x.shape[-1])
    factor = 2.0 ** (n_steps / 12.0)
    stretched = time_stretch(x, 1.0 / factor)  # 길이 변경(피치 변화)
    return circular_pad(stretched, n) if stretched.shape[-1] < n else stretched[..., :n]
