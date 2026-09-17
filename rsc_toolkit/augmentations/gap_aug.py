"""
GaP-Aug(Chang et al., ICASSP 2024) 의 스펙트로그램 증강 — **논문 명세 재구현** (공식·비공식 코드 없음, 등급 C).

논문 §2.2.2 / Algorithm 1 에서 확정된 것:
- 샘플(스펙트로그램)마다 무작위 패치 P 를 고르고 각 패치 p 에 p̃ = Gain·p^γ (gain 1) 를 적용.
- γ ~ U[1.7, 2.0], 패치 32개, 각 패치는 주파수 ≤16 × 시간 ≤16.
- Table 2 비교군: SpecAug(시간·주파수 구간 마스킹), PatchMask(패치를 0 으로).

가정(저자 확인 전):
- 비음수화: log-mel 은 음수라 거듭제곱이 정의되지 않는다. 기본 = 샘플별 min-max 로 [0,1] 정규화한 값에 γ 적용
  (정규화는 모델 쪽 `norm_mode` 가 담당; 이 모듈은 입력이 이미 비음수라고 가정하고 **패치 영역만** clamp(min=0) 후 거듭제곱한다 →
  norm_mode=none 이어도 패치 밖은 불변, 패치 안의 음수 값만 0 으로 소실).
  질문 #1 후반('정규화 후 모델 입력도 그 스케일인가?') 의 갈림길: 기본 sample_minmax 는 모델 입력도 [0,1];
  대안 sample_minmax_restore 는 γ 만 [0,1] 에서 적용하고 sample_minmax_split/sample_minmax_restore 로 원 dB 스케일로 되돌려 모델에 넣는다.
- 패치 높이·너비는 각각 1..16 에서 균등, 위치 균등, 겹침 허용(겹치면 순차 적용 → 지수 곱), 매 iteration 재추출.
- γ 는 패치마다 독립(gamma_per="patch"; "sample" 이면 샘플당 1개), 적용 확률 p=1.0.
- SpecAug 파라미터 미기재 → 백본이 PANNs 이므로 PANNs SpecAugmentation 기본값(time 64×2, freq 8×2) 가정, 마스크 값 0.
  주의(가정의 의미): 공식 Cnn14 는 bn0 **뒤** 에 SpecAugmentation 을 걸어 0 = BN 정규화 평균이지만, 여기서는 bn0 앞의 [0,1]
  정규화 입력에 걸어 0 = 샘플 최솟값(가장 조용한 값) 이 된다(GaP-Aug·PatchMask 와 같은 위치에서 비교하기 위함).
  time_drop_width 64 프레임은 PANNs hop 10 ms 기준 0.64 s 이며 paper preset(hop 6.25 ms) 에선 0.4 s 다 — 프레임 수 그대로 둔다.
  저자가 Transfer_Cnn14 를 그대로 썼다면 내장 spec_augmenter 가 Table 2 전 행(Naïve 포함) 에 켜져 있었을 가능성 → 저자 질문 후보.
- PatchMask = GaP-Aug 와 같은 패치 추출 규칙으로 패치를 0 으로 (논문 'randomly masks patches').

입력 규약: (B, 1, T, F) — PANNs 내부 배치 레이아웃(시간, mel). 높이=주파수 축(F), 너비=시간 축(T).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def sample_patches(
    n_patches: int, T: int, F: int, max_t: int, max_f: int, min_t: int, min_f: int, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    """(n_patches, 4) long: [t0, t1, f0, f1) — 크기 균등(min..max, 축 길이로 상한), 위치 균등."""
    max_t, max_f = max(1, min(max_t, T)), max(1, min(max_f, F))
    min_t, min_f = max(1, min(min_t, max_t)), max(1, min(min_f, max_f))
    ht = torch.randint(min_t, max_t + 1, (n_patches,), generator=gen)
    hf = torch.randint(min_f, max_f + 1, (n_patches,), generator=gen)
    t0 = (torch.rand(n_patches, generator=gen) * (T - ht + 1).float()).long()
    f0 = (torch.rand(n_patches, generator=gen) * (F - hf + 1).float()).long()
    return torch.stack([t0, t0 + ht, f0, f0 + hf], dim=1)


class GammaPatchAug(nn.Module):
    """
    p̃ = gain · p^γ 를 샘플별 n_patches 개 패치에 적용 (mode="gamma"), 또는 같은 패치를 0 으로 (mode="mask" = PatchMask).
    입력 (B,1,T,F) 는 비음수여야 한다 (norm_mode 로 [0,1] 정규화된 log-mel).
    """

    def __init__(
        self,
        n_patches: int = 32,
        max_freq: int = 16,
        max_time: int = 16,
        min_freq: int = 1,            # 가정: 최소 1
        min_time: int = 1,
        gamma_range: Tuple[float, float] = (1.7, 2.0),
        gain: float = 1.0,
        gamma_per: str = "patch",     # patch | sample (가정)
        p: float = 1.0,               # 샘플별 적용 확률 (가정 1.0)
        mode: str = "gamma",          # gamma | mask
    ):
        super().__init__()
        if mode not in ("gamma", "mask"):
            raise ValueError(f"mode must be gamma|mask, got {mode!r}")
        if gamma_per not in ("patch", "sample"):
            raise ValueError(f"gamma_per must be patch|sample, got {gamma_per!r}")
        self.n_patches, self.max_freq, self.max_time = int(n_patches), int(max_freq), int(max_time)
        self.min_freq, self.min_time = int(min_freq), int(min_time)
        self.gamma_lo, self.gamma_hi = float(gamma_range[0]), float(gamma_range[1])
        self.gain, self.gamma_per, self.p, self.mode = float(gain), gamma_per, float(p), mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, gen: Optional[torch.Generator] = None, return_info: bool = False):
        if x.dim() != 4:
            raise ValueError(f"expected (B,1,T,F), got {tuple(x.shape)}")
        B, _, T, F = x.shape
        out = x.clone()
        info = []
        for b in range(B):
            if self.p < 1.0 and torch.rand((), generator=gen).item() >= self.p:
                info.append(None)
                continue
            patches = sample_patches(self.n_patches, T, F, self.max_time, self.max_freq, self.min_time, self.min_freq, gen)
            if self.gamma_per == "sample":
                g = torch.empty(1).uniform_(self.gamma_lo, self.gamma_hi, generator=gen).expand(self.n_patches).clone()
            else:
                g = torch.empty(self.n_patches).uniform_(self.gamma_lo, self.gamma_hi, generator=gen)
            for k in range(self.n_patches):
                t0, t1, f0, f1 = (int(v) for v in patches[k])
                if self.mode == "mask":
                    out[b, 0, t0:t1, f0:f1] = 0.0
                else:
                    region = out[b, 0, t0:t1, f0:f1].clamp(min=0.0)  # 방어: 패치 영역만 비음수화 (norm_mode 가 [0,1] 로 맞춘다)
                    out[b, 0, t0:t1, f0:f1] = self.gain * region.pow(float(g[k]))
            info.append({"patches": patches, "gamma": g})
        return (out, info) if return_info else out


class SpecAugPANNs(nn.Module):
    """
    Table 2 'SpecAug' 비교군 — 파라미터 미기재라 PANNs SpecAugmentation 기본값 가정
    (time_drop_width 64 ×2, freq_drop_width 8 ×2; 각 stripe 폭 U[0, width), 값 0). torchaudio iid 마스킹으로 구현.
    적용 위치는 bn0 앞 [0,1] 입력(공식 Cnn14 는 bn0 뒤) — 모듈 docstring 참고(가정).
    입력 (B,1,T,F) → 내부에서 (B,1,F,T) 로 바꿔 torchaudio 규약(…, freq, time) 에 맞춘다.
    """

    def __init__(self, time_drop_width: int = 64, time_stripes: int = 2, freq_drop_width: int = 8, freq_stripes: int = 2, p: float = 1.0):
        super().__init__()
        from torchaudio import transforms as TA

        self.p = float(p)
        self.masks = nn.Sequential(
            *[TA.FrequencyMasking(freq_mask_param=freq_drop_width, iid_masks=True) for _ in range(freq_stripes)],
            *[TA.TimeMasking(time_mask_param=time_drop_width, iid_masks=True) for _ in range(time_stripes)],
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and torch.rand(()).item() >= self.p:
            return x
        return self.masks(x.transpose(2, 3)).transpose(2, 3).contiguous()


def normalize_nonneg(x: torch.Tensor, mode: str = "sample_minmax", db_offset: float = 100.0, eps: float = 1e-8) -> torch.Tensor:
    """
    log-mel 비음수화 (가정, 질문 #1). x: (B,1,T,F).
      sample_minmax : 샘플별 (x-min)/(max-min) → [0,1]   (기본 가정)
      batch_minmax  : 배치 전체 min-max → [0,1]
      db_offset     : (x + db_offset).clamp(min=0) / db_offset   (전역 dB 오프셋; PANNs log-mel 은 대략 [-100, +40] dB)
      none          : 그대로 (γ 보정은 패치 영역만 clamp(min=0) 후 적용 — 패치 안 음수 소실, 패치 밖 불변; 비교용)
      sample_minmax_restore : 평가 입력은 원 dB 그대로(= none). 학습 시 증강만 [0,1] 에서 하고 복원 → 모델 쪽 make_augmented_input 참고
    """
    if mode == "sample_minmax_restore":
        return x
    if mode == "sample_minmax":
        mn = x.amin(dim=(1, 2, 3), keepdim=True)
        mx = x.amax(dim=(1, 2, 3), keepdim=True)
        return (x - mn) / (mx - mn + eps)
    if mode == "batch_minmax":
        return (x - x.min()) / (x.max() - x.min() + eps)
    if mode == "db_offset":
        return (x + db_offset).clamp(min=0.0) / db_offset
    if mode == "none":
        return x
    raise ValueError(f"unknown norm mode {mode!r}")


def sample_minmax_split(x: torch.Tensor, eps: float = 1e-8):
    """샘플별 min-max: (x01, mn, mx) — sample_minmax_restore 용. x01 = (x-mn)/(mx-mn+eps)."""
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - mn) / (mx - mn + eps), mn, mx


def sample_minmax_restore(x01: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[0,1] → 원 dB 스케일 복원: x01·(mx-mn+eps) + mn (γ 로 줄어든 패치는 mn 쪽으로 내려간다)."""
    return x01 * (mx - mn + eps) + mn
