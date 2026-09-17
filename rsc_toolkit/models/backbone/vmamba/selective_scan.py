"""
Selective scan (Mamba S6 recurrence) — same interface as SelectiveScanOflex in VMamba/DASS `csms6s.py`.

Two paths:
- ``torch``: pure-torch reference implementation (slow but exact; CPU and GPU). Computes the recurrence along L,
  h_l = exp(Δ_l·A)·h_{l-1} + Δ_l·B_l·u_l, with a Hillis-Steele parallel scan, and y_l = C_l·h_l + D·u_l. The math is identical to mamba's
  selective_scan_ref (DASS kernels/selective_scan/test_selective_scan.py:168-236); it computes in fp32 and returns fp32
  (same output dtype as the oflex kernel with `oflex=True`).
- ``cuda_oflex``: the DASS/VMamba CUDA extension ``selective_scan_cuda_oflex`` (optional dependency; build kernels/selective_scan
  following the DASS README). Used by default when it can be imported and the input is on CUDA.

Selection: ``backend='auto'`` (default) → kernel if importable and the tensor is on CUDA, otherwise torch. Set the environment variable
RSC_SELECTIVE_SCAN=torch to force the fallback. Numerical agreement of the two paths is checked in tests/unit/models/test_dass_lungsrad.py.

Tensor conventions (as at the csms6s.forward_corev2 call site):
  u, delta: (B, K·D, L)   A: (K·D, N)   Bs, Cs: (B, K, N, L)   Ds, delta_bias: (K·D,)
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

try:  # optional dependency — falls back to pure torch when absent
    import selective_scan_cuda_oflex as _ss_oflex  # type: ignore
except Exception:  # pragma: no cover - when the kernel is unavailable
    _ss_oflex = None


def cuda_kernel_available() -> bool:
    return _ss_oflex is not None


def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_l = a_l·h_{l-1} + b_l (h_{-1}=0), scan axis = dim 2. a, b: (B, D, L, N). Hillis-Steele, ⌈log2 L⌉ steps."""
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
    # expand group B/C (B,K,N,L) to channels (B,K·D,N,L): group k covers d channels (with_Group in csms6s.selective_scan_ref)
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
    """Pure-torch path. oflex=True gives fp32 output, otherwise the input dtype. Uses activation checkpointing by default during training (when grad is required)."""
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled() and (u.requires_grad or delta.requires_grad)
    args = (u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus)
    if use_checkpoint:
        y = _ckpt.checkpoint(_selective_scan_torch_impl, *args, use_reentrant=False)
    else:
        y = _selective_scan_torch_impl(*args)
    return y if oflex else y.to(u.dtype)


class SelectiveScanOflex(torch.autograd.Function):
    """csms6s.SelectiveScanOflex as-is (CUDA extension selective_scan_cuda_oflex)."""

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
    """'auto' | 'torch' | 'cuda_oflex' → actual path name."""
    env = os.environ.get("RSC_SELECTIVE_SCAN", "").strip().lower()
    if env in {"torch", "cuda_oflex"}:
        backend = env
    if backend == "auto":
        return "cuda_oflex" if (_ss_oflex is not None and device.type == "cuda") else "torch"
    if backend == "cuda_oflex":
        if _ss_oflex is None:
            raise RuntimeError("selective_scan_cuda_oflex kernel cannot be imported (build kernels/selective_scan)")
        if device.type != "cuda":
            raise RuntimeError("the cuda_oflex path is only available for CUDA tensors")

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
