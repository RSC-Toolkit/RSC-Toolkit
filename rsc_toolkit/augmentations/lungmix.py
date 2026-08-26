"""
Lungmix(ICASSP 2025, Ge et al., arXiv 2501.00064) 파형 믹싱 증강.

출처:
- 논문: Lungmix: A Mixup-Based Strategy for Generalization in Respiratory Sound Classification
- 공식 코드: https://github.com/Tigerdwgth/ICASSP-2025-Lungmix lungmix.py (88줄, 라이선스 없음;
  클론 outputs/reference/lungmix/lungmix.py). 저장소에는 증강 함수만 있고 학습·데이터·모델·손실 코드가 없다.

두 갈래를 나란히 둔다:

(1) `lungmix_official()` — **P0 골든**: 공식 `lungmix()` 를 numpy/torch 전역 RNG 호출 순서까지 그대로 이식.
    같은 seed 에서 wav/label/lam/mask/max_length 가 비트 일치한다(scripts/reproduction/dump_lungmix_goldens.py,
    tests/unit/models/test_ast_lungmix.py). 공식 코드의 quirk 도 그대로다:
      - wav2 를 [0, max_len) 균등 roll 후 앞에 길이 shift2∈[0, len(wav1)) 의 **상수 mean(wav2) 패딩** → 재반복으로
        출력 길이가 max_len+shift2 (Fig.2 캡션의 'randomly generated padding' 과 다름).
      - λ~Beta(0.5, 0.5) (논문 미기재).
      - random mask 는 16 kHz **샘플 단위** i.i.d. Bernoulli(0.5).
      - loudness mask(|x|>|mean+2std| 의 OR)·λ·최종 mask{0,λ,1} 를 계산하지만 **출력 파형은 random mask 만으로 블렌딩**
        (lungmix.py:84-85) — 논문 식 3 의 M_Lungmix 블렌딩과 불일치. mask/lam 은 반환값으로만 나간다.
      - smooth(max_pool1d k=201) 옵션은 논문에 없다(기본 False).
      - 라벨 OR 는 문자열('normal'/'both') 기반.

(2) `lungmix_pair(..., apply_loudness_mask=True)` — **논문 식 3 재구현(플래그)**: 같은 RNG 순서로 같은 mask 를 만든 뒤
    x̃ = M ⊙ x_i + (1−M) ⊙ x_j, M∈{0,λ,1} 로 실제 블렌딩한다. 기본값 False 가 코드판(P0) 과 동일 출력이다.
    어느 쪽이 Table I 'non-linear' 행인지는 저자 확인 전(AUTHOR_QUESTIONS LungMix #1).

라벨 OR 는 4-class 인덱스 기반으로 일반화(`or_label`; Label Powerset, 논문 §II-C): normal 은 항등원, crackle|wheeze=both.

믹싱 입력 길이(가정, 저자 확인 전 — AUTHOR_QUESTIONS LungMix #5 첫 항목): 논문은 '9 s pad/cut' 과 믹싱의 순서를 명시하지 않았지만,
식 3 직후 마스크 차원을 M_Lungmix ∈ {0,λ,1}^{1×maxlen(x_i,x_j)} 로 적어(lungmix.txt:175) 두 입력의 길이가 **다르다**는 전제이고,
공식 함수의 첫 __repeat_audio_to_max_length 호출(lungmix.py:62)도 이미 9 s 로 맞춘 입력에는 항등이라 '원 구간(길이 상이)을 섞은 뒤
9 s 로 맞춤' 쪽 근거가 논문·코드 양쪽에 있다.
- mix_stage='raw' (**기본, 주 런**): 원 구간 길이(valid_len)만 잘라 섞은 뒤 9 s 로 repeat/cut 한다. lungmix_batch(valid_len=...) 경로
  (데이터셋이 패딩 전 길이를 배치 meta_labels 로 실어 준다).
- mix_stage='padded' (대안): 데이터셋이 이미 9 s repeat 패딩한 파형을 그대로 섞는다. 공식 함수 구조상 shift2∈[0, 9 s) 의
  상수 mean(wav2) 패딩이 붙어 cut_first 로 자르면 출력의 기대 25% 가 상수가 된다(Fig.2 'randomly generated padding' 과 다름).
  저자 확인 전이므로 둘 다 플래그.

마스크 겹침(가정, 저자 확인 전): 논문 식 8 은 M_Lungmix = M_Loudness ⊕ R (element-wise OR, lungmix.txt:203-208) 인데 공식 코드
(lungmix.py:82-83)는 random·loudness 가 겹치는 위치를 (1+λ)−1 = λ 로 둔다(OR 해석이면 1). apply_loudness_mask=True 경로의
`overlap='lam'`(기본) 은 공식 반환 mask 를 그대로 쓰고, `overlap='one'` 은 OR 해석(random 위치 = 1, loudness 만 = λ) 을 쓴다.
float32 반올림 때문에 공식 mask 에는 λ 가 두 값(λ, (1+λ)−1; 차이 ≤ 1e-7) 으로 존재한다 — 기록만.

짝짓기(가정, **플래그 없음·고정**): 배치 내 torch.randperm 짝, 믹싱 샘플이 원본을 대체(추가 아님). mix_prob 만 플래그.
vanilla_mixup_batch 의 λ 는 배치 공통 1개(가정).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

# ----------------------------------------------------------------------------- 공식 코드 그대로 (P0)


def _get_aug_label(a, b):
    """공식 __get__aug__label 그대로 (문자열 4-class 전용)."""
    if a == "normal":
        return b
    elif b == "normal":
        return a
    elif a != b:
        return "both"
    else:
        return a


def _repeat_audio_to_max_length(wav1, wav2):
    """공식 __repeat_audio_to_max_length 그대로."""
    max_length = max(wav1.shape[1], wav2.shape[1])
    combined_audio = torch.FloatTensor(torch.zeros(1, max_length))
    wav1 = wav1.repeat(1, math.ceil(max_length / wav1.shape[1]))[:, :max_length]
    wav2 = wav2.repeat(1, math.ceil(max_length / wav2.shape[1]))[:, :max_length]
    return wav1, wav2, max_length, combined_audio


def lungmix_official(wav1, wav2, label1, label2, smooth=False):
    """
    공식 lungmix(wav1, wav2, label1, label2, smooth) 의 비트 동일 이식 (전역 np.random / torch RNG 사용, 호출 순서 동일).
    반환: (wav, label, lam, label1, label2, wav1, wav2, mask, max_length) — 공식과 같은 9-튜플.
    """
    wav1, wav2, max_length, combined_audio = _repeat_audio_to_max_length(wav1, wav2)
    shift = int(max_length * np.random.rand())
    wav2 = wav2.roll(int(shift), dims=1)
    shift2 = int(wav1.shape[1] * np.random.rand())
    lam = np.random.beta(0.5, 0.5)
    wav2 = torch.cat([torch.mean(wav2) * torch.ones(1, shift2), wav2], dim=1)
    wav1, wav2, max_length, combined_audio = _repeat_audio_to_max_length(wav1, wav2)
    loudness_mask1 = torch.abs(wav1) > torch.abs((torch.mean(wav1) + 2 * torch.std(wav1)))
    loudness_mask2 = torch.abs(wav2) > torch.abs((torch.mean(wav2) + 2 * torch.std(wav2)))
    loudness_mask = loudness_mask1 | loudness_mask2
    if smooth:
        ma_windowsize = 201
        loudness_mask = torch.nn.functional.max_pool1d(
            loudness_mask.float(), kernel_size=ma_windowsize, stride=1, padding=(ma_windowsize - 1) // 2
        )
    random_mask = torch.rand(1, max_length) > 0.5
    loudness_mask = loudness_mask * lam
    mask = random_mask + loudness_mask
    mask = mask - (mask > 1).float()
    combined_audio[:, : wav1.shape[1]] += wav1[:, :]
    combined_audio = (random_mask) * combined_audio + (~random_mask) * wav2[:, :]
    wav = combined_audio
    label = _get_aug_label(label1, label2)
    return wav, label, lam, label1, label2, wav1, wav2, mask, max_length


# ----------------------------------------------------------------------------- 라벨 OR (인덱스 일반화)


@dataclass(frozen=True)
class ClassLayout:
    """4-class 인덱스 배치. 툴킷 ICBHI 기본: normal 0, crackle 1, wheeze 2, both 3."""

    normal: int = 0
    crackle: int = 1
    wheeze: int = 2
    both: int = 3

    def to_bits(self, y: int) -> int:
        return {self.normal: 0b00, self.crackle: 0b01, self.wheeze: 0b10, self.both: 0b11}[int(y)]

    def from_bits(self, bits: int) -> int:
        return {0b00: self.normal, 0b01: self.crackle, 0b10: self.wheeze, 0b11: self.both}[int(bits)]


DEFAULT_LAYOUT = ClassLayout()


def or_label(y_a: int, y_b: int, layout: ClassLayout = DEFAULT_LAYOUT) -> int:
    """
    논문 식 4 / §II-C Label Powerset: 비정상 비트(crackle, wheeze)의 OR. normal 은 항등원.
    공식 __get__aug__label 과 4-class 위에서 동일한 표(normal+X→X, X+X→X, crackle+wheeze→both, both+X→both).
    """
    return layout.from_bits(layout.to_bits(y_a) | layout.to_bits(y_b))


def or_labels(y_a: torch.Tensor, y_b: torch.Tensor, layout: ClassLayout = DEFAULT_LAYOUT) -> torch.Tensor:
    out = [or_label(int(a), int(b), layout) for a, b in zip(y_a.tolist(), y_b.tolist())]
    return torch.as_tensor(out, dtype=torch.long, device=y_a.device)


# ----------------------------------------------------------------------------- 파형 믹싱 (코드판 / 논문판)


@dataclass
class LungmixResult:
    wav: torch.Tensor            # (1, L_out), L_out = max_len + shift2 (공식 quirk; 호출측에서 cut)
    lam: float
    mask: torch.Tensor           # (1, L_out) float ∈ {0, λ, 1} — 공식 반환 mask 와 동일
    random_mask: torch.Tensor    # (1, L_out) bool
    loudness_mask: torch.Tensor  # (1, L_out) float (λ 곱 전, smooth 반영)
    wav1: torch.Tensor           # 처리 후 wav1 (1, L_out)
    wav2: torch.Tensor           # roll·패딩·재반복 후 wav2 (1, L_out)
    max_length: int
    blend_mask: Optional[torch.Tensor] = None  # 식 3 블렌딩에 실제 쓴 마스크 (apply_loudness_mask=False 면 random_mask 와 무관한 공식 mask)


def lungmix_pair(
    wav1: torch.Tensor,
    wav2: torch.Tensor,
    smooth: bool = False,
    apply_loudness_mask: bool = False,
    overlap: str = "lam",
) -> LungmixResult:
    """
    한 쌍의 파형을 Lungmix 로 섞는다. RNG 호출 순서·mask 계산은 lungmix_official 과 동일.

    apply_loudness_mask=False (기본, P0): 공식 코드와 같이 출력 = random_mask*wav1 + (~random_mask)*wav2.
    apply_loudness_mask=True (논문 식 3 재구현): 출력 = M*wav1 + (1-M)*wav2, M∈{0,λ,1}.
      overlap='lam'(기본): M = 공식 반환 mask (random∧loudness 겹침 = (1+λ)−1 = λ).
      overlap='one'      : M = 식 8 OR 해석 (random 위치 = 1, loudness 만 = λ, 나머지 0).
    반환 mask 는 항상 공식 반환 mask (골든 비교용); 블렌딩에 실제 쓴 마스크는 blend_mask.
    """
    if overlap not in OVERLAP_MODES:
        raise ValueError(f"overlap 은 {OVERLAP_MODES} 중 하나: {overlap!r}")
    if wav1.dim() == 1:
        wav1 = wav1.unsqueeze(0)
    if wav2.dim() == 1:
        wav2 = wav2.unsqueeze(0)
    wav1, wav2, max_length, combined_audio = _repeat_audio_to_max_length(wav1, wav2)
    shift = int(max_length * np.random.rand())
    wav2 = wav2.roll(int(shift), dims=1)
    shift2 = int(wav1.shape[1] * np.random.rand())
    lam = float(np.random.beta(0.5, 0.5))
    wav2 = torch.cat([torch.mean(wav2) * torch.ones(1, shift2), wav2], dim=1)
    wav1, wav2, max_length, combined_audio = _repeat_audio_to_max_length(wav1, wav2)
    loudness_mask1 = torch.abs(wav1) > torch.abs((torch.mean(wav1) + 2 * torch.std(wav1)))
    loudness_mask2 = torch.abs(wav2) > torch.abs((torch.mean(wav2) + 2 * torch.std(wav2)))
    loudness_mask = loudness_mask1 | loudness_mask2
    if smooth:
        ma_windowsize = 201
        loudness_mask = torch.nn.functional.max_pool1d(
            loudness_mask.float(), kernel_size=ma_windowsize, stride=1, padding=(ma_windowsize - 1) // 2
        )
    random_mask = torch.rand(1, max_length) > 0.5
    loudness_lam = loudness_mask * lam
    mask = random_mask + loudness_lam
    mask = mask - (mask > 1).float()          # {0, λ, 1}: random 만 → 1, loudness 만 → λ, 둘 다 → λ(1+λ−1)
    blend_mask = mask.float()
    if apply_loudness_mask:
        if overlap == "one":                  # 식 8 OR 해석: random 위치는 1
            blend_mask = torch.where(random_mask, torch.ones_like(loudness_lam), loudness_lam).float()
        wav = blend_mask * wav1 + (1.0 - blend_mask) * wav2            # 논문 식 3
    else:
        combined_audio[:, : wav1.shape[1]] += wav1[:, :]
        wav = (random_mask) * combined_audio + (~random_mask) * wav2[:, :]   # 공식 코드 84-85
    return LungmixResult(
        wav=wav, lam=lam, mask=mask.float(), random_mask=random_mask,
        loudness_mask=loudness_mask.float(), wav1=wav1, wav2=wav2, max_length=max_length, blend_mask=blend_mask,
    )


FIT_MODES = ("cut_first", "cut_last", "random_crop", "none")
OVERLAP_MODES = ("lam", "one")


def _repeat_to(wav: torch.Tensor, out_len: int) -> torch.Tensor:
    L = wav.shape[-1]
    if L >= out_len:
        return wav
    reps = -(-out_len // L)
    return wav.repeat(*([1] * (wav.dim() - 1)), reps)[..., :out_len]


def fit_length(wav: torch.Tensor, out_len: int, mode: str = "cut_first") -> torch.Tensor:
    """
    믹싱 출력(길이 max_len+shift2)을 학습 길이로 맞춘다. 짧으면 repeat 패딩.
    가정(저자 확인 전, AUTHOR_QUESTIONS LungMix #5): 기본 cut_first(앞 out_len). 대안 cut_last(뒤 out_len; shift2 상수 패딩을
    가장 많이 버린다), random_crop(무작위 시작, np.random 소비), none(그대로).
    """
    if mode == "none":
        return wav
    if mode not in FIT_MODES:
        raise ValueError(f"unknown fit mode {mode!r} (choose from {FIT_MODES})")
    wav = _repeat_to(wav, out_len)
    L = wav.shape[-1]
    if L == out_len or mode == "cut_first":
        return wav[..., :out_len]
    if mode == "cut_last":
        return wav[..., L - out_len:]
    start = int(np.random.randint(0, L - out_len + 1))
    return wav[..., start:start + out_len]


@dataclass
class LungmixBatch:
    wav: torch.Tensor      # (B, out_len)
    y_or: torch.Tensor     # (B,) OR 라벨
    y_a: torch.Tensor      # (B,) 첫 입력 라벨
    y_b: torch.Tensor      # (B,) 짝 라벨
    lam: torch.Tensor      # (B,) float — 쌍마다 다른 λ
    index: torch.Tensor    # (B,) 짝 인덱스


def lungmix_batch(
    wavs: torch.Tensor,
    labels: torch.Tensor,
    index: Optional[torch.Tensor] = None,
    out_len: Optional[int] = None,
    smooth: bool = False,
    apply_loudness_mask: bool = False,
    layout: ClassLayout = DEFAULT_LAYOUT,
    fit_mode: str = "cut_first",
    valid_len: Optional[torch.Tensor] = None,
    overlap: str = "lam",
) -> LungmixBatch:
    """
    배치 (B, L) 를 샘플별로 lungmix_pair 에 넣는다 (i 와 index[i] 를 섞는다; 결과 = 샘플별 함수 결과와 동일).
    index 가 None 이면 torch.randperm(B) (가정, 플래그 없음: 배치 내 permutation 짝, 원본 대체 — AUTHOR_QUESTIONS #4;
    확률은 호출측 mix_prob).
    valid_len (B,) 이 주어지면 각 파형의 앞 valid_len[i] 샘플(원 구간, 패딩 전)만 믹싱에 넣는다 (mix_stage='raw', 기본·주 런);
    None 이면 (B, L) 전체(이미 패딩된 파형, mix_stage='padded' 대안).
    RNG 는 전역(np.random / torch) 을 그대로 쓰므로 CPU 텐서로 호출해야 공식 함수와 같은 난수 열이 된다.
    """
    if wavs.dim() == 3:
        wavs = wavs.squeeze(1)
    B, L = wavs.shape
    out_len = int(out_len or L)
    if index is None:
        index = torch.randperm(B)
    index = index.to(wavs.device)
    if valid_len is not None:
        vl = [max(1, min(int(v), L)) for v in torch.as_tensor(valid_len).reshape(-1).tolist()]
        if len(vl) != B:
            raise ValueError(f"valid_len 길이 {len(vl)} != 배치 {B}")
    else:
        vl = [L] * B
    outs, lams = [], []
    for i in range(B):
        j = int(index[i])
        r = lungmix_pair(wavs[i : i + 1, : vl[i]], wavs[j : j + 1, : vl[j]], smooth=smooth,
                         apply_loudness_mask=apply_loudness_mask, overlap=overlap)
        outs.append(fit_length(r.wav, out_len, fit_mode).squeeze(0))
        lams.append(r.lam)
    y_a = labels
    y_b = labels[index]
    return LungmixBatch(
        wav=torch.stack(outs, 0), y_or=or_labels(y_a, y_b, layout), y_a=y_a, y_b=y_b,
        lam=torch.as_tensor(lams, dtype=torch.float32, device=wavs.device), index=index,
    )


def vanilla_mixup_batch(
    wavs: torch.Tensor, labels: torch.Tensor, index: Optional[torch.Tensor] = None, alpha: float = 0.5,
) -> LungmixBatch:
    """
    Table I 'vanilla Mixup' 비교군 — 파형 선형 mixup x̃ = λ x_i + (1−λ) x_j, 배치 공통 λ~Beta(α,α).
    가정(논문 미기재): α=0.5(공식 lungmix 의 Beta(0.5,0.5) 와 맞춤), 파형 단계 믹싱. y_or 는 OR 라벨(비교용).
    """
    if wavs.dim() == 3:
        wavs = wavs.squeeze(1)
    B = wavs.shape[0]
    if index is None:
        index = torch.randperm(B)
    index = index.to(wavs.device)
    lam = float(np.random.beta(alpha, alpha))
    wav = lam * wavs + (1.0 - lam) * wavs[index]
    return LungmixBatch(
        wav=wav, y_or=or_labels(labels, labels[index]), y_a=labels, y_b=labels[index],
        lam=torch.full((B,), lam, dtype=torch.float32, device=wavs.device), index=index,
    )


__all__ = [
    "ClassLayout", "DEFAULT_LAYOUT", "FIT_MODES", "OVERLAP_MODES", "LungmixBatch", "LungmixResult", "fit_length", "lungmix_batch",
    "lungmix_official", "lungmix_pair", "or_label", "or_labels", "vanilla_mixup_batch",
]
