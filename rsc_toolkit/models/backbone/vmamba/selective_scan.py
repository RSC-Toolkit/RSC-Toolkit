"""
Selective scan (Mamba S6 recurrence) — VMamba/DASS `csms6s.py` 의 SelectiveScanOflex 와 같은 인터페이스.

두 경로:
- ``torch``: 순수 torch 참조 구현(느리지만 정확; CPU/GPU 모두). Hillis-Steele 병렬 스캔으로 L 방향 순환
  h_l = exp(Δ_l·A)·h_{l-1} + Δ_l·B_l·u_l 을 계산하고 y_l = C_l·h_l + D·u_l. 수식은 mamba 의 selective_scan_ref
  (DASS kernels/selective_scan/test_selective_scan.py:168-236) 와 동일하며, fp32 로 계산해 fp32 를 돌려준다
  (oflex 커널의 `oflex=True` 출력 dtype 과 동일).
- ``cuda_oflex``: DASS/VMamba 의 CUDA 확장 ``selective_scan_cuda_oflex`` (선택 의존성; kernels/selective_scan 빌드,
  README §커널 빌드 참고). import 되고 입력이 CUDA 면 기본으로 이 경로를 쓴다.

선택: ``backend='auto'`` (기본) → 커널 import 가능·CUDA 텐서면 커널, 아니면 torch. 환경변수 RSC_SELECTIVE_SCAN=torch 로 강제 폴백.
두 경로의 수치 일치는 tests/unit/models/test_dass_lungsrad.py 에서 검사한다.

텐서 규약 (csms6s.forward_corev2 호출부 그대로):
  u, delta: (B, K·D, L)   A: (K·D, N)   Bs, Cs: (B, K, N, L)   Ds, delta_bias: (K·D,)
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

try:  # 선택 의존성 — 없으면 순수 torch 폴백
    import selective_scan_cuda_oflex as _ss_oflex  # type: ignore
except Exception:  # pragma: no cover - 커널이 없을 때
    _ss_oflex = None


def cuda_kernel_available() -> bool:
    return _ss_oflex is not None


def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_l = a_l·h_{l-1} + b_l (h_{-1}=0), 스캔 축 = dim 2. a, b: (B, D, L, N). Hillis-Steele, ⌈log2 L⌉ 단계."""
    L = a.shape[2]
    s = 1
    while s < L:
        b = torch.cat([b[:, :, :s], b[:, :, s:] + a[:, :, s:] * b[:, :, :-s]], dim=2)
        a = torch.cat([a[:, :, :s], a[:, :, s:] * a[:, :, :-s]], dim=2)
        s *= 2
    return b


def _selective_scan_torch_impl(u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus: bool) -> torch.Tensor:
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    Bsz, KD, L = u.shape
    N = A.shape[1]
    K = Bs.shape[1]
    d = KD // K
    A = A.float()
    # 그룹 B/C (B,K,N,L) → 채널로 확장 (B,K·D,N,L) : 그룹 k 가 d 개 채널을 담당 (csms6s.selective_scan_ref 의 with_Group)
    Bx = Bs.float().unsqueeze(2).expand(Bsz, K, d, N, L).reshape(Bsz, KD, N, L)
    Cx = Cs.float().unsqueeze(2).expand(Bsz, K, d, N, L).reshape(Bsz, KD, N, L)
    deltaA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])              # (B, KD, L, N)
    deltaB_u = (delta * u).unsqueeze(-1) * Bx.permute(0, 1, 3, 2)               # (B, KD, L, N)
    h = _parallel_scan(deltaA, deltaB_u)                                         # (B, KD, L, N)
    y = (h * Cx.permute(0, 1, 3, 2)).sum(-1)                                     # (B, KD, L)
    if Ds is not None:
        y = y + u * Ds.float()[None, :, None]
    return y


def selective_scan_torch(u, delta, A, Bs, Cs, Ds=None, delta_bias=None, delta_softplus=True,
                         oflex: bool = True, use_checkpoint: Optional[bool] = None) -> torch.Tensor:
    """순수 torch 경로. oflex=True 면 fp32 출력, 아니면 입력 dtype. 학습 중(grad 필요) 기본으로 activation checkpoint."""
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled() and (u.requires_grad or delta.requires_grad)
    args = (u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus)
    if use_checkpoint:
        y = _ckpt.checkpoint(_selective_scan_torch_impl, *args, use_reentrant=False)
    else:
        y = _selective_scan_torch_impl(*args)
    return y if oflex else y.to(u.dtype)


class SelectiveScanOflex(torch.autograd.Function):
    """csms6s.SelectiveScanOflex 그대로 (CUDA 확장 selective_scan_cuda_oflex)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = _ss_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = _ss_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def resolve_backend(backend: str, device: torch.device) -> str:
    """'auto' | 'torch' | 'cuda_oflex' → 실제 경로 이름."""
    env = os.environ.get("RSC_SELECTIVE_SCAN", "").strip().lower()
    if env in {"torch", "cuda_oflex"}:
        backend = env
    if backend == "auto":
        return "cuda_oflex" if (_ss_oflex is not None and device.type == "cuda") else "torch"
    if backend == "cuda_oflex":
        if _ss_oflex is None:
            raise RuntimeError("selective_scan_cuda_oflex 커널이 import 되지 않는다 (kernels/selective_scan 빌드 필요)")
        if device.type != "cuda":
            raise RuntimeError("cuda_oflex 경로는 CUDA 텐서에서만 쓸 수 있다")
        return "cuda_oflex"
    if backend == "torch":
        return "torch"
    raise ValueError(f"unknown selective-scan backend {backend!r}")


def selective_scan(u, delta, A, Bs, Cs, Ds=None, delta_bias=None, delta_softplus=True, oflex=True,
                   backend: str = "auto") -> torch.Tensor:
    name = resolve_backend(backend, u.device)
    if name == "cuda_oflex":
        return SelectiveScanOflex.apply(u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus, -1, -1, oflex)
    return selective_scan_torch(u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus, oflex)
