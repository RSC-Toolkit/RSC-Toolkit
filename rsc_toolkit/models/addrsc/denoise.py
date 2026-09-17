# Portions of this file are adapted from AFNO-transformer (https://github.com/NVlabs/AFNO-transformer), class AFNO1D; that code is under the CC-BY-NC-SA-4.0 license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Adaptive differential denoising layer of ADD-RSC (Interspeech 2025, deegy666/ADD-RSC) — faithful port of
models/adapt_diff_denoise.py (exact port of the official code, verified against goldens).

DiffTransformerLayer(d_model 256, num_heads 8, depth 6) is a **single layer** (`depth` is only used to compute λ_init = 0.8 − 0.6·exp(−0.3·depth)).
forward(x (B, N=1024 frames, C=256 frequency bins)):
  x1 = AFNO1D(x)                                  # time-axis rfft(ortho) → block-diagonal 2-layer MLP (complex, ReLU) → softshrink(0.01) → irfft + residual
  x2 = classifier((x1[:,0] + x1[:,1]) / 2)        # auxiliary logits: **mean of the first two frames** → Linear(256, 4) (hard-coded in the original)
  y  = attn(norm1(x1)) + x1                       # differential attention (RMSNorm pre-norm)
  z  = ff(norm2(y)) + y                           # SwiGLU
  return z, x2
The original MultiHeadDifferentialAttention.forward builds `nn.LayerNorm(N).to('cuda:0')` on every call to normalize the attention but
never uses the result (dead code; LayerNorm init consumes no RNG) — it is not created here. Otherwise the init order (kaiming/xavier) matches the original.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class AFNO1D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.hidden_size, self.sparsity_threshold, self.num_blocks = hidden_size, sparsity_threshold, num_blocks
        self.block_size = hidden_size // num_blocks
        self.hard_thresholding_fraction, self.hidden_size_factor = hard_thresholding_fraction, hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(torch.empty(2, num_blocks, self.block_size, self.block_size * hidden_size_factor))
        self.b1 = nn.Parameter(torch.zeros(2, num_blocks, self.block_size * hidden_size_factor))
        self.w2 = nn.Parameter(torch.empty(2, num_blocks, self.block_size * hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(torch.zeros(2, num_blocks, self.block_size))
        nn.init.kaiming_uniform_(self.w1, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w2, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)
        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)
        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        o1_real[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[1]) + self.b1[0])
        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[1]) + self.b1[1])
        o2_real[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[1]) + self.b2[0])
        o2_imag[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[1]) + self.b2[1])
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        return x.type(dtype) + bias


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        return self.W2(F.silu(self.WG(x)) * self.W1(x))


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads, depth):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)
        self.lambda_q1 = nn.Parameter(torch.empty(num_heads, self.d_head))
        nn.init.kaiming_uniform_(self.lambda_q1, a=0, mode="fan_in", nonlinearity="relu")
        self.lambda_k1 = nn.Parameter(torch.empty(num_heads, self.d_head))
        nn.init.kaiming_uniform_(self.lambda_k1, a=0, mode="fan_in", nonlinearity="relu")
        self.lambda_q2 = nn.Parameter(torch.empty(num_heads, self.d_head))
        nn.init.kaiming_uniform_(self.lambda_q2, a=0, mode="fan_in", nonlinearity="relu")
        self.lambda_k2 = nn.Parameter(torch.empty(num_heads, self.d_head))
        nn.init.kaiming_uniform_(self.lambda_k2, a=0, mode="fan_in", nonlinearity="relu")
        self.lambda_init = lambda_init_fn(depth)
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X):
        batch, N, d_model = X.shape
        Q = self.W_q(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = self.W_k(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = self.W_v(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling
        attention = F.softmax(A1, dim=-1) - lambda_val * F.softmax(A2, dim=-1)
        # (original: builds a new nn.LayerNorm(N) here to normalize attention, but the result is unused — omitted)
        O = torch.matmul(attention, V)
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head) * (1 - self.lambda_init)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)
        return self.W_o(O_concat)


class DiffTransformerLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, depth=6, n_cls=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, depth)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)
        self.classifier = nn.Linear(256, n_cls)  # original hard-codes Linear(256, 4) — creation order preserved
        self.aff = AFNO1D(hidden_size=d_model)

    def forward(self, x):
        x1 = self.aff(x)
        x2 = self.classifier((x1[:, 0] + x1[:, 1]) / 2)
        y = self.attn(self.norm1(x1)) + x1
        z = self.ff(self.norm2(y)) + y
        return z, x2


class LabelSmoothingLoss(nn.Module):
    """models/bias_denoise_loss.py — label smoothing based on KLDiv(reduction none), normalized by the number of valid targets."""

    def __init__(self, size=4, padding_idx=5, smoothing=0.20, normalize_length=True):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x, target):
        batch_size = x.size(0)
        x = x.contiguous().view(-1, self.size)
        target = target.contiguous().view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
