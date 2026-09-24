"""
Disentanglement loss components of DDE-MAE (Wei·Miao·Li, Interspeech 2025, arXiv 2506.10698): **re-implemented from the paper specification**
(no public code → no goldens; equation numbers follow the paper body).

- time_shuffle          : paper §2.2 "time-shuffled" input X̃ (random shuffle along the time axis). The unit (frame / patch column) is not specified →
                          default 'patch' (shuffle the order of the 64 patch columns of 16 frames each), alternative 'frame' (shuffle 1024 frames); assumption flag.
                          Rationale for the 'patch' default: the physically same patch can be masked in both inputs (shuffle_mask=same_physical) and tokens correspond.
                          The literal reading of "randomly shuffling the time axis of the spectrogram" is 'frame' (stronger shuffle, forces same_index);
                          checking sensitivity to both values is recommended.
- siamese_hinge         : Eq. 6  L_siamese = mean_i( max( ‖E_I(X)_i − E_I(X̃)_i‖₂ − margin, 0 ) ). margin not specified → 1.0 assumed.
- VCLUB                 : Eq. 4  vCLUB upper bound (Cheng et al., ICML 2020)  I(Z_R;Z_I) ≤ E_p(z_R,z_I)[log q_θ(z_I|z_R)] − E_p(z_R)p(z_I)[log q_θ(z_I|z_R)],
                          q_θ(z_I|z_R) = N(μ_θ(z_R), diag σ²_θ(z_R)) (re-implements the MLP form of the official CLUB implementation CLUB.py; whether the input is per token
                          or a pooled vector is not specified → pooled vector (mean of patch tokens) assumed). q_θ must be trained separately by maximizing log-likelihood, so
                          the model uses an internal auxiliary Adam (lr not specified → 1e-4 assumed, one update per step); see DDEMAE in rsc_toolkit/models/audiomae_dde.py.

All of the above are assumptions not confirmed by the authors.
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
    x: (B, 1, T, F), Audio-MAE input batch (T=1024 frames, F=128 mel). Independent permutation per sample.
    unit='patch': shuffle the order of the T/patch_frames patch columns (16 frames each) (default assumption).
    unit='frame': shuffle the T frames individually (alternative).
    Returns (x̃, perm) with x̃[b, :, t, :] = x[b, :, perm[b, t], :] (indices in units of `unit`).
    """
    B, C, T, Fdim = x.shape
    if unit == "patch":
        if T % patch_frames != 0:
            raise ValueError(f"T={T} is not a multiple of patch_frames={patch_frames}")
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
    """Inverse of time_shuffle (for tests and verification)."""
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
    Permutes the masking noise (B, grid_t*grid_f; t-major) according to the patch-column permutation perm (B, grid_t):
    column t' of the shuffled input X̃ is original column perm[t'], so noise_s[:, t', f] = noise[:, perm[t'], f].
    → the mask built by argsort keeps **the physically same patches** in X and X̃ (assumption: 'same mask' means same patches).
    """
    B = noise.shape[0]
    n2 = noise.view(B, grid_t, grid_f)
    idx = perm.view(B, grid_t, 1).expand(B, grid_t, grid_f)
    return torch.gather(n2, 1, idx).reshape(B, grid_t * grid_f)


# ------------------------------------------------------------------------------------------ Siamese hinge (Eq. 6)
def siamese_hinge(z_a: torch.Tensor, z_b: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    z_a, z_b: (B, D) pooled vectors (assumption) or (B, N, D) tokens; L2 distance over the last dim, mean over the rest.
    L = mean( max( ‖z_a − z_b‖₂ − margin, 0 ) )
    """
    d = torch.linalg.vector_norm(z_a - z_b, ord=2, dim=-1)
    return F.relu(d - margin).mean()


# ------------------------------------------------------------------------------------------ vCLUB (Eq. 4)
class VCLUB(nn.Module):
    """
    Variational CLUB upper-bound estimator. q_θ(z_I | z_R) = N(μ(z_R), diag exp(logσ²(z_R))).
    Structure (assumption, form of the 'CLUB' class in the official CLUB.py): μ: Linear(x,h)-ReLU-Linear(h,y), logσ²: Linear(x,h)-ReLU-Linear(h,y)-Tanh.

    - loglikelihood(z_r, z_i) : E[log q_θ(z_i|z_r)] (without constants), for training (maximizing) q_θ.
    - mi_upper_bound(z_r, z_i): Eq. 4 = positive − negative,
        positive = mean_i  −(μ(z_r_i) − z_i_i)² / 2σ²_i          (∑_d)
        negative = mean_{i,j} −(μ(z_r_i) − z_i_j)² / 2σ²_i        (∑_d)
      (the log σ term depends only on i and cancels, so it is omitted as in the CLUB implementation.)
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
        Computes Eq. 4 with the q_θ parameters held constant (gradients flow only to z_r, z_i),
        so that the main loss backward does not touch q_θ (q_θ is updated only by the auxiliary Adam).
        """
        mu_params = {k: v.detach() for k, v in self.p_mu.named_parameters()}
        lv_params = {k: v.detach() for k, v in self.p_logvar.named_parameters()}
        mu = torch.func.functional_call(self.p_mu, mu_params, (z_r,))
        logvar = torch.func.functional_call(self.p_logvar, lv_params, (z_r,))
        positive = -((mu - z_i) ** 2) / 2.0 / logvar.exp()
        negative = -((z_i.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
