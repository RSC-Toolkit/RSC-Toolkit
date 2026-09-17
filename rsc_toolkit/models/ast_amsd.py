"""
AMS-D — Attentive Masked Self-Distillation for Respiratory Sound Classification (Nuo Chen, Mingsheng Xu; ICASSP 2026).
Faithful port of the official code CcnNnn/AMS-D 5f8aab9 (MIT) models/{ams_d,ast_model,token_weight_module}.py + main.py train()
(exact port of the official code, verified against goldens).

Architecture (following the official CLI defaults — the paper does not specify some details such as hard/soft masking):
- AST base384, fshape/tshape = fstride/tstride = 16, input 128×798 (freq×time) → 8×49 = 392 tokens, no axis-swap quirk.
- The AudioSet 16/16 ckpt (audioset_16_16_0.4422.pth) is loaded in full into an inner 128×1024 (8×64, pos_embed 514) model, then
  pos_embed is center-cropped along time [32-24 : 32-24+49] = [8:57] (ams_d.py:198-211; the `int(f_dim/4)` typo in the f_dim<8 branch is
  kept as well — never executed since f_dim=8).
  Note: the toolkit ckpt is HF MIT/ast-finetuned-audioset-16-16-0.442 converted to the YuanGong key format (pretrained_models/upstream/*.provenance.json,
    within 5e-5 of the HF logits) — no original YuanGong file with the authors' file name exists.
- mask_embed (1,1,768) xavier_normal (ams_d.py:184-185); hard masking (mask_dense 0 = main.py CLI default; the constructor default 0.5 is overridden by the CLI).
- num_masked = round(0.39·392) = 153. gpredlayer 768→768→ReLU→256 regresses the masked tokens (+2 offset) of the v.norm output onto the input 16×16 patches (Unfold) → MSE.
- TokenWeights (token_weight_module.py): x[:,2:] detach → Conv1d k3 → LN → Linear→1 → softmax (token axis) weighted sum → LN+Linear auxiliary logits;
  next-epoch mask = top-153 weights (descending).
- Classification = (cls+dist)/2 → mlp_head (LN+Linear). Inference (eval/validation) uses only the AST path, without masking or the auxiliary head (main.py:393).

Training state (main.py:180-187, 321-350): per-sample all_predictions (N×4, softmax probabilities; one-hot at epoch 0) and ids_masked (N×153) are
**register_buffer**s, so they are saved/restored automatically by EpochResumeCheckpoint (last.ckpt). Sample indices arrive via meta_labels
(datasets/amsd_data.py attaches the official train index order (IndexDataset df row number) as the 4th element). The current epoch is the buffer current_epoch
(set by trainer/callbacks/amsd_epoch.py at every on_train_epoch_start).

Loss = rsc_toolkit/methods/amsd.py AMSDLoss (Custom_Loss, double-softmax teacher kept).

method flag (paper Table 3 / Table 2):
  amsd (default) = CE + α·KL + β·CE_aux + γ·MSE, attentive masking      (Table 3 row 4)
  sd_mask    = CE + α·KL + β·CE_aux, attentive masking, no reconstruction (Table 3 row 3; Assumption: identical except γ=0)
  sd_only    = CE + α·KL, no masking/auxiliary/reconstruction            (Table 3 row 2 = Table 2 'No Masking'; variant absent from the code → assumption)
  baseline   = CE only (AST full-FT)                                     (Table 3 row 1; Assumption: same data pipeline and noise/roll. per-sample buffers are not updated)
mask_mode: attentive | random | time_inter | freq_inter (Table 2; as in ams_d.py:245-252 — non-attentive modes mask and compute MSE from epoch 0,
  no auxiliary head (weighted_output None → β term 0, main.py flow); 'cluster' is dead code and not ported).

Noise/roll at epoch ≥ noise_start_epoch(10) (mydataset.py:126-129): **after** normalization, add torch.manual_seed(2345+epoch+index) U(0,1)·(np.random.rand()/10)
to the fbank + time-axis roll randint(-10,10). The torch part is reproduced seed-exactly with a CPU generator; amplitude/roll use the worker-global
numpy RNG in the original and cannot be reproduced → (Assumption) numpy Generator derived from (noise_seed, epoch, index). The official code also
applies this to the valid subset, but the toolkit does not apply it to val (= official test).

For input preprocessing (clip length) see datasets/amsd_data.py — the default is the 8 s loop-pad of paper §3.1 (**assumption**); the fbank zero-padding path is a second assumption.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ast_patchmix import PatchEmbed, _create_deit_skeleton
from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)

AUDIOSET_16_16_CANDIDATES = (
    "pretrained_models/upstream/audioset_16_16_0.4422.pth",
    "pretrained_models/audioset_16_16_0.4422.pth",
)
METHODS = ("amsd", "sd_mask", "sd_only", "baseline")
MASK_MODES = ("attentive", "random", "time_inter", "freq_inter")
_MASK_ALIASES = {"time": "time_inter", "freq": "freq_inter"}  # no 'none' — Table 2 'No Masking' is method=sd_only


class TokenWeights(nn.Module):
    """token_weight_module.py TokenWeights as-is (the f_dim/t_dim kwargs are unused in the original too)."""

    def __init__(self, len_masked: int, embed_dim: int, dim_out: int, num_classes: int, groups: int = 1):
        super().__init__()
        assert embed_dim == dim_out, "embed_dim should be the same with dim_out"
        self.len_masked = len_masked
        self.dim_out = dim_out
        self.num_classes = num_classes
        self.conv = nn.Conv1d(embed_dim, dim_out, kernel_size=3, padding=1, bias=False, groups=groups)
        self.norm = nn.LayerNorm(dim_out)
        self.score = nn.Linear(dim_out, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(dim_out), nn.Linear(dim_out, num_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_f = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_f = self.norm(x_f)
        token_score = self.score(x_f)
        token_weight = torch.softmax(token_score, dim=1)
        ids_shuffle = torch.argsort(token_weight.squeeze(-1), dim=1, descending=True)
        ids_masked = ids_shuffle[:, : self.len_masked]
        weighted_features = torch.sum(x * token_weight, dim=1)
        weighted_x = self.classifier(weighted_features)
        return ids_masked, weighted_x


def _resolve_ckpt(pretrained_path: Optional[str]) -> pathlib.Path:
    from rsc_toolkit.paths import resolve_pretrained

    if pretrained_path is not None:
        p = resolve_pretrained(explicit=pretrained_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    found = resolve_pretrained(AUDIOSET_16_16_CANDIDATES)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"audioset_16_16_0.4422.pth not found (create it with scripts/reproduction/convert_hf_ast_to_yuangong.py): {AUDIOSET_16_16_CANDIDATES}"
    )


@ModelFactory.register_model_class("ast_amsd")
class AMSDAST(BaseRSCModel, nn.Module):
    accepts_meta = ("amsd_sample_index",)  # training batch[2] = sample index (AMSDDataModule amsd_return_index)

    def __init__(
        self,
        n_cls: int = 4,
        input_fdim: int = 128,
        input_tdim: int = 798,
        fshape: int = 16,
        tshape: int = 16,
        fstride: int = 16,
        tstride: int = 16,
        model_size: str = "base384",
        pretrained_path: Optional[str] = None,
        mask_ratio: float = 0.39,
        mask_dense: float = 0.0,          # main.py --mask-dense default 0 (hard masking)
        mask_mode: str = "attentive",
        method: str = "amsd",
        num_train_samples: int = 4142,     # number of train cycles in the official 60/40 split (buffer size; the callback corrects it to the datamodule length)
        # loss (main.py --alpha/--beta/--gamma/--tau, Custom_Loss distillation_type='soft')
        alpha: float = 1.0,
        beta: float = 3e-2,
        gamma: float = 3e-1,
        tau: float = 1.0,
        distillation_type: str = "soft",
        teacher_is_prob: bool = False,     # assumption flag: True removes the double softmax (methods/amsd.py)
        # epoch ≥10 noise/roll (mydataset.py:126-129)
        noise_enabled: bool = True,
        noise_start_epoch: int = 10,
        noise_seed: int = 2345,
        noise_roll_range: Tuple[int, int] = (-10, 10),
        input_layout: str = "freq_time",   # toolkit fbank (B,128,798); "time_freq" for original-layout input
        verbose: bool = False,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        mask_mode = _MASK_ALIASES.get(mask_mode, mask_mode)
        if mask_mode not in MASK_MODES:
            hint = " ('No Masking' is method=sd_only, not a mask_mode)" if str(mask_mode).lower() in ("none", "no", "off") else ""
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}{hint}")
        if model_size != "base384":
            raise ValueError("currently only has base384 AudioSet pretrained model.")  # ams_d.py:163-164
        if input_layout not in ("freq_time", "time_freq"):
            raise ValueError(f"input_layout must be freq_time|time_freq, got {input_layout!r}")
        self.n_cls = int(n_cls)
        self.method = method
        self.mask_mode = mask_mode
        self.mask_ratio = float(mask_ratio)
        self.mask_dense = float(mask_dense)
        self.input_layout = input_layout
        self.noise_enabled = bool(noise_enabled)
        self.noise_start_epoch = int(noise_start_epoch)
        self.noise_seed = int(noise_seed)
        self.noise_roll_range = (int(noise_roll_range[0]), int(noise_roll_range[1]))
        self.fshape, self.tshape, self.fstride, self.tstride = fshape, tshape, fstride, tstride
        self.unfold = nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

        # ---- inner ASTModel(128×1024, 16/16, imagenet=False) + full 16/16 ckpt load (ams_d.py:174-177)
        self.v = _create_deit_skeleton(model_size, pretrained=False)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        inner_f, inner_t = self._conv_grid(128, 1024)
        inner_patches = inner_f * inner_t  # 8 × 64 = 512
        patch_embed = PatchEmbed(img_size=(128, 1024), patch_size=(fshape, tshape), in_chans=1, embed_dim=self.original_embedding_dim)
        patch_embed.proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        patch_embed.num_patches = inner_patches
        self.v.patch_embed = patch_embed
        self.v.pos_embed = nn.Parameter(torch.zeros(1, inner_patches + 2, self.original_embedding_dim))
        self.audioset_ckpt = self._load_audioset(pretrained_path)

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, n_cls))
        self.gpredlayer = nn.Sequential(
            nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256)
        )
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
        torch.nn.init.xavier_normal_(self.mask_embed)

        f_dim, t_dim = self._conv_grid(input_fdim, input_tdim)
        self.f_dim, self.t_dim = f_dim, t_dim
        self.num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = self.num_patches
        self.num_masked = round(self.mask_ratio * self.num_patches)
        if verbose:
            logger.info("AMSDAST: f_dim=%d t_dim=%d patches=%d masked=%d", f_dim, t_dim, self.num_patches, self.num_masked)

        # ams_d.py:198-211 as-is (constants 512/8/64/32/4, including the int(f_dim/4) typo in the f_dim<8 branch)
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 512, 768).transpose(1, 2).reshape(1, 768, 8, 64)
        if t_dim < 64:
            new_pos_embed = new_pos_embed[:, :, :, 32 - int(t_dim / 2) : 32 - int(t_dim / 2) + t_dim]
        elif t_dim > 64:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode="bilinear")
        if f_dim < 8:
            new_pos_embed = new_pos_embed[:, :, 4 - int(f_dim / 2) : 4 - int(f_dim / 4) + f_dim, :]
        elif f_dim > 8:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
        new_pos_embed = new_pos_embed.reshape(1, 768, self.num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        self.token_weight = TokenWeights(len_masked=self.num_masked, embed_dim=768, dim_out=768, num_classes=n_cls)

        from rsc_toolkit.methods.amsd import AMSDLoss

        self.criterion = AMSDLoss(distillation_type=distillation_type, tau=tau, alpha=alpha, beta=beta, gamma=gamma,
                                  teacher_is_prob=teacher_is_prob)

        # ---- PS-KD state buffers (main.py:180-187) — saved to/restored from the ckpt
        self.register_buffer("all_predictions", torch.zeros(int(num_train_samples), n_cls))
        self.register_buffer("ids_masked", torch.zeros(int(num_train_samples), self.num_masked, dtype=torch.long))
        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ construction
    def _conv_grid(self, input_fdim: int, input_tdim: int) -> Tuple[int, int]:
        test_proj = nn.Conv2d(1, 1, kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
        out = test_proj(torch.randn(1, 1, input_fdim, input_tdim))
        return out.shape[2], out.shape[3]

    def _load_audioset(self, pretrained_path: Optional[str]) -> str:
        """ams_d.py:173-177: DataParallel(ASTModel).load_state_dict(sd, strict=False), then take .module.v = load all module.v.* keys."""
        p = _resolve_ckpt(pretrained_path)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        v_state = {}
        for k, val in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("v."):
                v_state[k[len("v."):]] = val
        missing, unexpected = self.v.load_state_dict(v_state, strict=False)
        real_missing = [k for k in missing if not k.startswith(("head", "head_dist"))]
        real_unexpected = [k for k in unexpected if not k.startswith(("head", "head_dist"))]
        if real_missing or real_unexpected:
            raise RuntimeError(f"AudioSet 16/16 ckpt load differs from the original. missing={real_missing[:5]} unexpected={real_unexpected[:5]}")
        return str(p)

    def ensure_buffers(self, num_train_samples: int) -> None:
        """Resize buffers to the actual number of train samples (no-op after a resume load, where they already match)."""
        n = int(num_train_samples)
        if self.all_predictions.shape[0] == n:
            return
        logger.warning("Reallocating AMSD buffers %d → %d (num_train_samples argument does not match the datamodule length)", self.all_predictions.shape[0], n)
        dev = self.all_predictions.device
        self.all_predictions = torch.zeros(n, self.n_cls, device=dev)
        self.ids_masked = torch.zeros(n, self.num_masked, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------ masking (ams_d.py:299-438)
    def _apply_mask(self, x: torch.Tensor, ids_masked: torch.Tensor, dense: float) -> torch.Tensor:
        B, N, D = x.shape
        mask_dense = torch.ones([B, N, D], device=x.device, dtype=x.dtype)
        mask_dense.scatter_(1, ids_masked.unsqueeze(-1).expand(-1, -1, D), dense)
        mask_tokens = self.mask_embed.expand(B, N, D)
        return x * mask_dense + (1 - mask_dense) * mask_tokens

    def attentive_mask(self, x, ids_masked):
        return self._apply_mask(x, ids_masked, self.mask_dense), ids_masked

    def random_mask_patch(self, x, num_masked):
        B, N, D = x.shape
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)
        ids_masked = ids_shuffle[:, :num_masked]
        return self._apply_mask(x, ids_masked, self.mask_dense), ids_masked

    def time_interval_masking(self, x, t_dim, f_dim, num_masked):
        B, N, D = x.shape
        assert N == t_dim * f_dim
        num_mask_cols = max(1, num_masked // f_dim)
        num_mask_remain = num_masked - num_mask_cols * f_dim
        n_cols = num_mask_cols + 1 if num_mask_remain > 0 else num_mask_cols
        col_indices = torch.linspace(0, t_dim - 1, n_cols, dtype=torch.long, device=x.device)
        mask_indices = []
        for j, col in enumerate(col_indices):
            if j < num_mask_cols:
                mask_indices.extend([int(col) + t_dim * t for t in range(f_dim)])
            else:
                mask_indices.extend([int(col_indices[-1]) + t_dim * t for t in range(num_mask_remain)])
        ids_masked = torch.tensor(mask_indices, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        return self._apply_mask(x, ids_masked, 0.0), ids_masked  # original is fixed to hard (0)

    def freq_interval_masking(self, x, t_dim, f_dim, num_masked):
        B, N, D = x.shape
        assert N == t_dim * f_dim
        num_mask_rows = max(1, num_masked // t_dim)
        num_mask_remains = num_masked - num_mask_rows * t_dim
        n_rows = num_mask_rows + 1 if num_mask_remains > 0 else num_mask_rows
        row_indices = torch.linspace(0, f_dim - 1, n_rows, dtype=torch.long, device=x.device)
        mask_indices = []
        for j, row in enumerate(row_indices):
            start_idx = int(row) * t_dim
            if j < num_mask_rows:
                mask_indices.extend(range(start_idx, start_idx + t_dim))
            else:
                mask_indices.extend(range(start_idx, start_idx + num_mask_remains))
        ids_masked = torch.tensor(mask_indices, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        return self._apply_mask(x, ids_masked, 0.0), ids_masked

    # ------------------------------------------------------------------ forward
    def _to_paper_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Toolkit (B,128,798) → original (B,798,128) = (time, freq)."""
        if x.dim() == 4:
            x = x.squeeze(1)
        if self.input_layout == "freq_time":
            return x.transpose(-2, -1)
        return x

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, tokens), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        return self.v.norm(x)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_head((x[:, 0] + x[:, 1]) / 2)

    def forward_paper(self, x_tf: torch.Tensor) -> torch.Tensor:
        """Original forward(mode='eval')[0] — x_tf: (B,798,128)."""
        inp = x_tf.unsqueeze(1).transpose(2, 3)
        return self._head(self._encode(self.v.patch_embed(inp)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_paper(self._to_paper_layout(x))

    def amsd_forward(self, x_tf: torch.Tensor, epoch: int, ids_masked: Optional[torch.Tensor],
                     use_mask: bool = True, use_aux: bool = True, use_recon: bool = True):
        """Original AMSD.forward(mode='train') (ams_d.py:226-296). Returns (weighted_x|None, mse, ids_masked|None, s_out)."""
        inp = x_tf.unsqueeze(1).transpose(2, 3)  # (B,1,128,798)
        x = self.v.patch_embed(inp)
        B, N, D = x.shape
        masked_now = False
        if use_mask:
            if self.mask_mode == "attentive":
                if epoch > 0:
                    x, ids_masked = self.attentive_mask(x, ids_masked)
                    masked_now = True
            elif self.mask_mode == "random":
                x, ids_masked = self.random_mask_patch(x, self.num_masked)
                masked_now = True
            elif self.mask_mode == "time_inter":
                x, ids_masked = self.time_interval_masking(x, self.t_dim, self.f_dim, self.num_masked)
                masked_now = True
            elif self.mask_mode == "freq_inter":
                x, ids_masked = self.freq_interval_masking(x, self.t_dim, self.f_dim, self.num_masked)
                masked_now = True
        x = self._encode(x)

        # reconstruction MSE (ams_d.py:263-277): attentive has no masking at epoch 0 → mse 0
        if use_recon and masked_now:
            patches = self.unfold(inp).transpose(1, 2)  # (B,392,256) — same order as the conv patches (verified against goldens)
            gidx = ids_masked.unsqueeze(-1)
            pred = self.gpredlayer(torch.gather(x, 1, (gidx + 2).expand(-1, -1, D)))
            target = torch.gather(patches, 1, gidx.expand(-1, -1, patches.shape[-1]))
            mse = torch.mean((pred - target) ** 2)
        else:
            mse = x.new_zeros(())

        weighted_x = None
        if use_aux and self.mask_mode == "attentive":
            detach_x_patch = x[:, 2:, :].detach()
            ids_masked, weighted_x = self.token_weight(detach_x_patch)
        s_out = self._head(x)
        return weighted_x, mse, ids_masked, s_out

    # ------------------------------------------------------------------ noise/roll (mydataset.py:126-129, after normalization)
    def _noise_roll(self, x_tf: torch.Tensor, epoch: int, sample_idx: torch.Tensor) -> torch.Tensor:
        out = []
        T, Fq = x_tf.shape[1], x_tf.shape[2]
        lo, hi = self.noise_roll_range
        for i in range(x_tf.shape[0]):
            n = int(sample_idx[i])
            g = torch.Generator().manual_seed(self.noise_seed + epoch + n)          # torch.manual_seed(2345+epoch+index) (CPU stream)
            u = torch.rand(T, Fq, generator=g).to(x_tf.device, x_tf.dtype)
            rng = np.random.default_rng([self.noise_seed, epoch, n])                # Assumption: original uses the worker-global numpy RNG
            amp = float(rng.random()) / 10                                          # np.random.rand()/10
            shift = int(rng.integers(lo, hi))                                       # np.random.randint(-10, 10)
            out.append(torch.roll(x_tf[i] + u * amp, shift, 0))
        return torch.stack(out, 0)

    # ------------------------------------------------------------------ training
    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        x = self._to_paper_layout(audio_batch)
        if self.training and meta_labels is None:
            raise ValueError("ast_amsd training batch has no sample index (meta_labels) — use the icbhi_amsd data config with data.params.amsd_return_index=true (otherwise it silently trains a baseline)")
        if not self.training:
            # validation (main.py:393,423): AST-path logits + CE
            logits = self.forward_paper(x)
            return logits, loss_func(logits, labels)

        epoch = int(self.current_epoch)
        if not getattr(self, "_epoch_callback_attached", False) and not getattr(self, "_epoch_warned", False):
            # Training without AMSDEpochCallback (callbacks=amsd_paper*) leaves epoch at 0 forever:
            # attentive masking and noise/roll stay disabled and the teacher is overwritten with one-hot
            # at every step, so self-distillation degrades to CE+KL(one-hot) — without any error.
            # (Missing meta_labels is caught by the raise above, but a stuck epoch only changes results silently.)
            # Some paths (e.g. tests) drive the model directly without the callback, so warn once instead of raising.
            logger.warning(
                "ast_amsd is training without AMSDEpochCallback — the epoch is stuck at 0, so "
                "masking/noise never turn on and distillation degrades. Use a callbacks=amsd_paper* "
                "config (when driving the model directly, update the model.current_epoch buffer)."
            )
            self._epoch_warned = True
        idx = meta_labels.long()
        if self.noise_enabled and epoch >= self.noise_start_epoch:
            x = self._noise_roll(x, epoch, idx)

        if self.method == "baseline":
            logits = self.forward_paper(x)
            return logits, loss_func(logits, labels)

        use_mask = self.method in ("amsd", "sd_mask")
        use_aux = use_mask
        use_recon = self.method == "amsd"

        onehot = F.one_hot(labels, self.n_cls).to(x.dtype)
        if epoch == 0:
            self.all_predictions[idx] = onehot                                     # main.py:321-322
        t_outputs = self.all_predictions[idx]                                      # main.py:328
        ids_in = self.ids_masked[idx] if (use_mask and self.mask_mode == "attentive") else None
        weighted_x, mse, ids_out, s_out = self.amsd_forward(x, epoch, ids_in, use_mask=use_mask, use_aux=use_aux, use_recon=use_recon)
        loss, _, _, _ = self.criterion(s_out, t_outputs, weighted_x, onehot, mse)  # main.py:333
        with torch.no_grad():                                                      # main.py:348-350
            self.all_predictions[idx] = F.softmax(s_out.detach(), dim=1).to(self.all_predictions.dtype)
            if use_mask and self.mask_mode == "attentive" and ids_out is not None:
                self.ids_masked[idx] = ids_out.detach()
        return s_out, loss
