"""
DDE-MAE(Wei·Miao·Li, Interspeech 2025, arXiv 2506.10698) 의 분리(disentanglement) 손실 부품 — **논문 명세 재구현**
(공개 코드 없음 → 골든 없음. 식 번호는 논문 본문 기준).

- time_shuffle          : 논문 §2.2 "time-shuffled" 입력 X̃ (시간축 무작위 셔플). 단위(프레임/패치 열)는 미기재 →
                          기본 'patch'(16 프레임 = 패치 열 64개의 순서 셔플), 대안 'frame'(1024 프레임 셔플) — 가정 플래그.
                          'patch' 를 기본으로 둔 근거: 두 입력에서 물리적으로 같은 패치를 마스킹(shuffle_mask=same_physical)·토큰 대응이 가능.
                          논문 문구 "randomly shuffling the time axis of the spectrogram" 의 문자적 해석은 'frame'(더 강한 셔플, same_index 강제)
                          → 저자 질문 7번 답 전까지 사다리 후 두 값 민감도 확인 권장.
- siamese_hinge         : 식 6  L_siamese = mean_i( max( ‖E_I(X)_i − E_I(X̃)_i‖₂ − margin, 0 ) ). margin 미기재 → 1.0 가정.
- VCLUB                 : 식 4  vCLUB 상한 (Cheng et al., ICML 2020)  I(Z_R;Z_I) ≤ E_p(z_R,z_I)[log q_θ(z_I|z_R)] − E_p(z_R)p(z_I)[log q_θ(z_I|z_R)],
                          q_θ(z_I|z_R) = N(μ_θ(z_R), diag σ²_θ(z_R)) (CLUB 공식 구현 CLUB.py 의 MLP 형태 재구현; 입력이 토큰별인지
                          풀링 벡터인지 미기재 → 풀링 벡터(패치 토큰 평균) 가정). q_θ 는 log-likelihood 최대화로 따로 학습해야 하므로
                          모델 내부 보조 Adam(lr 미기재 → 1e-4 가정, 매 스텝 1회) — rsc_toolkit/models/audiomae_dde.py DDEMAE 참조.

전부 '가정(저자 확인 전)' — docs/AUTHOR_QUESTIONS_2026-08-26.md DDE-MAE 절 4·5·7 번.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------ time-shuffle
def time_shuffle(
    x: torch.Tensor,
    unit: str = "patch",
    patch_frames: int = 16,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, 1, T, F) — Audio-MAE 입력 배치 (T=1024 프레임, F=128 mel). 샘플마다 독립 순열.
    unit='patch': T/patch_frames 개의 패치 열(각 16 프레임) 순서를 섞는다 (기본 가정).
    unit='frame': T 프레임을 프레임 단위로 섞는다 (대안).
    반환: (x̃, perm) — x̃[b, :, t, :] = x[b, :, perm[b, t], :] (unit 단위 인덱스).
    """
    B, C, T, Fdim = x.shape
    if unit == "patch":
        if T % patch_frames != 0:
            raise ValueError(f"T={T} 가 patch_frames={patch_frames} 의 배수가 아니다")
        n = T // patch_frames
        perm = torch.stack([torch.randperm(n, generator=generator) for _ in range(B)]).to(x.device)  # (B, n)
        xs = x.reshape(B, C, n, patch_frames, Fdim)
        idx = perm.view(B, 1, n, 1, 1).expand(B, C, n, patch_frames, Fdim)
        xs = torch.gather(xs, 2, idx)
        return xs.reshape(B, C, T, Fdim), perm
    if unit == "frame":
        perm = torch.stack([torch.randperm(T, generator=generator) for _ in range(B)]).to(x.device)  # (B, T)
        idx = perm.view(B, 1, T, 1).expand(B, C, T, Fdim)
        return torch.gather(x, 2, idx), perm
    raise ValueError(f"unit must be 'patch' or 'frame', got {unit!r}")


def time_unshuffle(x_shuffled: torch.Tensor, perm: torch.Tensor, unit: str = "patch", patch_frames: int = 16) -> torch.Tensor:
    """time_shuffle 의 역연산 (테스트·검증용)."""
    B, C, T, Fdim = x_shuffled.shape
    inv = torch.argsort(perm, dim=1)
    if unit == "patch":
        n = T // patch_frames
        xs = x_shuffled.reshape(B, C, n, patch_frames, Fdim)
        idx = inv.view(B, 1, n, 1, 1).expand(B, C, n, patch_frames, Fdim)
        return torch.gather(xs, 2, idx).reshape(B, C, T, Fdim)
    idx = inv.view(B, 1, T, 1).expand(B, C, T, Fdim)
    return torch.gather(x_shuffled, 2, idx)


def permute_token_noise(noise: torch.Tensor, perm: torch.Tensor, grid_t: int, grid_f: int) -> torch.Tensor:
    """
    마스킹 noise (B, grid_t*grid_f; t-major) 를 패치 열 순열 perm (B, grid_t) 에 맞춰 옮긴다:
    셔플 입력 X̃ 의 열 t' 는 원본 열 perm[t'] 이므로 noise_s[:, t', f] = noise[:, perm[t'], f].
    → argsort 로 만든 마스크가 X 와 X̃ 에서 **물리적으로 같은 패치** 를 남긴다 (가정: '같은 마스크' 를 같은 패치로 해석).
    """
    B = noise.shape[0]
    n2 = noise.view(B, grid_t, grid_f)
    idx = perm.view(B, grid_t, 1).expand(B, grid_t, grid_f)
    return torch.gather(n2, 1, idx).reshape(B, grid_t * grid_f)


# ------------------------------------------------------------------------------------------ Siamese hinge (식 6)
def siamese_hinge(z_a: torch.Tensor, z_b: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    z_a, z_b: (B, D) 풀링 벡터 (가정) 또는 (B, N, D) 토큰 — 마지막 차원 L2 거리, 나머지 차원 평균.
    L = mean( max( ‖z_a − z_b‖₂ − margin, 0 ) )
    """
    d = torch.linalg.vector_norm(z_a - z_b, ord=2, dim=-1)
    return F.relu(d - margin).mean()


# ------------------------------------------------------------------------------------------ vCLUB (식 4)
class VCLUB(nn.Module):
    """
    변분 CLUB 상한 추정기. q_θ(z_I | z_R) = N(μ(z_R), diag exp(logσ²(z_R))).
    구조(가정, CLUB 공식 CLUB.py 'CLUB' 클래스 형태): μ: Linear(x,h)-ReLU-Linear(h,y), logσ²: Linear(x,h)-ReLU-Linear(h,y)-Tanh.

    - loglikelihood(z_r, z_i) : E[log q_θ(z_i|z_r)] (상수항 제외) — q_θ 학습(최대화)용.
    - mi_upper_bound(z_r, z_i): 식 4 = positive − negative,
        positive = mean_i  −(μ(z_r_i) − z_i_i)² / 2σ²_i          (∑_d)
        negative = mean_{i,j} −(μ(z_r_i) − z_i_j)² / 2σ²_i        (∑_d)
      (log σ 항은 i 에만 의존해 상쇄되므로 CLUB 구현처럼 생략.)
    """

    def __init__(self, x_dim: int = 768, y_dim: int = 768, hidden: int = 512):
        super().__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, y_dim), nn.Tanh())

    def get_mu_logvar(self, z_r: torch.Tensor):
        return self.p_mu(z_r), self.p_logvar(z_r)

    def loglikelihood(self, z_r: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.get_mu_logvar(z_r)
        return (-((mu - z_i) ** 2) / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def mi_upper_bound(self, z_r: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.get_mu_logvar(z_r)
        positive = -((mu - z_i) ** 2) / 2.0 / logvar.exp()  # (B, D)
        prediction_1 = mu.unsqueeze(1)   # (B, 1, D)
        y_samples_1 = z_i.unsqueeze(0)   # (1, B, D)
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()  # (B, D)
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def mi_upper_bound_detached_q(self, z_r: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        """
        q_θ 파라미터를 상수로 두고(그래디언트는 z_r, z_i 로만) 식 4 를 계산한다 —
        주 손실의 backward 가 q_θ 를 건드리지 않게 하기 위함 (q_θ 는 보조 Adam 만 갱신).
        """
        mu_params = {k: v.detach() for k, v in self.p_mu.named_parameters()}
        lv_params = {k: v.detach() for k, v in self.p_logvar.named_parameters()}
        mu = torch.func.functional_call(self.p_mu, mu_params, (z_r,))
        logvar = torch.func.functional_call(self.p_logvar, lv_params, (z_r,))
        positive = -((mu - z_i) ** 2) / 2.0 / logvar.exp()
        negative = -((z_i.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
