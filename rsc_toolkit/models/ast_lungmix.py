"""
AST classifier of Lungmix (ICASSP 2025, Ge et al.) — **re-implementation from the paper description** (the official repository has no model or training code).

Paper §III-B: ImageNet+AudioSet-pretrained AST, pooling by 'mean of all hidden states instead of the first two tokens (cls/dist)', BatchNorm before
the classifier, input 128-mel × 1024 frames (9 s waveform), AudioSet mean/std normalization. During training, waveform Lungmix (online) within
the batch, then fbank.

Implementation: inherits rsc_toolkit.models.ast_patchmix.PatchMixAST (AudioSet 10/10 ckpt loading, forward skeleton, patch_mix), but
restores the **original AST input layout (input_fdim=128, input_tdim=1024)** → the AudioSet pos_embed (12×101) is used as-is without interpolation
(the swapped (798,128) quirk of patch-mix is not inherited — nothing in the paper indicates use of the patch-mix codebase).

Assumptions (all exposed as flags):
- stride not specified → fstride=tstride=10 (AudioSet 10/10 ckpt). Whether cls/dist are included in the all-token mean is not specified → pool='mean_all' (included);
     'mean_patches' excludes them. 'All hidden states' is assumed to be the tokens **after** the final LayerNorm (v.norm) (same as the HF
     last_hidden_state definition; mean before the norm is not implemented). BatchNorm = a single BatchNorm1d(768).
     head not specified: the original AST classifier is LayerNorm(768)+Linear, but the paper only says 'BatchNorm added before the classifier' →
     assumed head='bn_linear' (BN **replaces** LN); 'bn_ln_linear' is BN → LayerNorm → Linear (original mlp_head kept + BN added).
     SpecAugment not mentioned → specaugment.enabled=False (when enabled, toolkit SpecAugment after fbank, policy icbhi_ast_sup — the data config's
     specaugment block has no effect with kind: waveform, so it is handled here).
- 1024 frames: 9 s@10 ms ≈ 898 frames → the rest is zero-padded. Padding/normalization order not specified → fbank_pad_before_norm=True
     (original AST dataloader order: pad, then (x−mean)/(2·std)).
- Pairing: in-batch randperm pairs; the mixed sample **replaces** the original — **no flag (fixed)**; only the mixing probability is a flag, mix_prob=1.0.
- Length after mixing (max_len+shift2) → repeat, then cut the first 9 s (fit_mode='cut_first'; alternatives cut_last|random_crop). At the raw stage the
     output is ≤ 2× the original segment and usually shorter than 9 s → repeat padding dominates and fit_mode differences are small.
- Mixing stage: the notation M ∈ {0,λ,1}^{1×maxlen(x_i,x_j)} in paper Eq. 3 (two inputs of different length) and the length-matching structure of the
     official function (identity for equal-length inputs) suggest 'mix the original segments, then fit to 9 s' → mix_stage='raw' (**default, main run**):
     mixes only the original segment lengths carried in the batch meta_labels (requires data.params.lungmix.emit_valid_len=true), then fits to 9 s.
     mix_stage='padded' (alternative): mixes the waveforms the dataset already repeat-padded to 9 s (due to the constant shift2 padding in the official
     function, an expected 25% of the output is the constant mean(wav2)). Unconfirmed by the authors.
- apply_loudness_mask=False is the official code output (random mask only) — default (matches the official code). True re-implements paper Eq. 3.
     Mask overlap value on the Eq. 3 path: overlap='lam' (mask returned by the official code, random∧loudness = λ) default / 'one' (Eq. 8 OR interpretation = 1) alternative.
- 'Lungmix + Patch-mix': waveform Lungmix → patch_mix on the patch embeddings (β=1, original patch-mix) → labels follow the assumed patchmix_label_mode:
     'mixup' = PatchMixLoss(y_or, y_or[index], λ) linear interpolation / 'preserve' = CE(y_or) (combined with the label-preservation Patch-mix of §III-C).
     The paper does not state which one the combined version uses (comparison variant; not a reproduction target).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from rsc_toolkit.augmentations.lungmix import OVERLAP_MODES, ClassLayout, lungmix_batch, vanilla_mixup_batch
from rsc_toolkit.methods.lungmix import VARIANTS, LungmixLoss
from rsc_toolkit.processors.mel import generate_fbank

from .ast_patchmix import PatchMixAST
from .factory import ModelFactory

logger = logging.getLogger(__name__)

AUDIOSET_MEAN, AUDIOSET_STD = -4.2677393, 4.5689974


@ModelFactory.register_model_class("ast_lungmix")
class ASTLungmix(PatchMixAST):
    accepts_meta = ("lungmix_valid_len",)  # training batch[2] = original segment length (LungmixDataModule emit_valid_len)

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        # ---- pooling / head (paper §III-B) ----
        pool: str = "mean_all",            # mean_all (mean of all tokens incl. cls/dist) | mean_patches | cls_dist (original AST)
        use_bn: bool = True,
        head: str = "bn_linear",           # bn_linear (Assumption: BN replaces the original LayerNorm) | bn_ln_linear (BN → LN → Linear)
        specaugment: Optional[dict] = None,  # {enabled: false, policy: icbhi_ast_sup, mask: mean} — not mentioned in the paper → off by default
        # ---- online fbank ----
        sample_rate: int = 16000,
        n_mels: int = 128,
        target_frames: int = 1024,
        fbank_pad_before_norm: bool = True,
        fbank_mean: float = AUDIOSET_MEAN,
        fbank_std: float = AUDIOSET_STD,
        # ---- mixing / loss ----
        variant: str = "non_linear",
        apply_loudness_mask: bool = False,
        overlap: str = "lam",              # mask overlap on the Eq. 3 path: lam (mask returned by the official code) | one (Eq. 8 OR interpretation)
        smooth: bool = False,
        mix_prob: float = 1.0,
        out_len_sec: float = 9.0,
        fit_mode: str = "cut_first",
        mix_stage: str = "raw",            # raw (default, main run; requires meta_labels = original segment length) | padded (alternative)
        vanilla_alpha: float = 0.5,
        lambda1: float = 1.0,
        lambda2_mode: str = "ratio",
        lambda2: float = 1.0,
        mix_beta: float = 1.0,             # patch_mix Beta(β,β) for lungmix_patchmix
        patchmix_label_mode: str = "mixup",  # lungmix_patchmix: mixup (PatchMixLoss linear interpolation) | preserve (CE(y_or))
        time_domain: bool = False,
        class_layout: Optional[dict] = None,
        mix_num_threads: int = 1,
    ):
        super().__init__(
            n_cls=n_cls, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain, model_size=model_size,
            mix_beta=mix_beta, pretrained_path=pretrained_path, verbose=verbose, method="ce",
            time_domain=time_domain, input_layout="time_freq",
        )
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}: {variant!r}")
        if pool not in ("mean_all", "mean_patches", "cls_dist"):
            raise ValueError("pool must be mean_all|mean_patches|cls_dist")
        if head not in ("bn_linear", "bn_ln_linear"):
            raise ValueError("head must be bn_linear|bn_ln_linear")
        if mix_stage not in ("padded", "raw"):
            raise ValueError("mix_stage must be padded|raw")
        if overlap not in OVERLAP_MODES:
            raise ValueError(f"overlap must be one of {OVERLAP_MODES}")
        if patchmix_label_mode not in ("mixup", "preserve"):
            raise ValueError("patchmix_label_mode must be mixup|preserve")
        # the parent's mlp_head/classifier (LayerNorm+Linear) is unused → excluded from the parameter count
        self.mlp_head = nn.Identity()
        self.classifier = nn.Identity()
        self.pool = pool
        self.head_mode = head
        self.bn = nn.BatchNorm1d(self.original_embedding_dim) if use_bn else nn.Identity()
        self.ln = nn.LayerNorm(self.original_embedding_dim) if head == "bn_ln_linear" else nn.Identity()
        self.head = nn.Linear(self.original_embedding_dim, n_cls)
        self.fstride, self.tstride = int(fstride), int(tstride)
        sa = dict(specaugment or {})
        self.specaugment = None
        if sa.get("enabled", False):
            from rsc_toolkit.augmentations.specaugment import SpecAugment

            self.specaugment = SpecAugment(policy=sa.get("policy", "icbhi_ast_sup"), mask=sa.get("mask", "mean"))

        self.sample_rate, self.n_mels, self.target_frames = int(sample_rate), int(n_mels), int(target_frames)
        self.fbank_pad_before_norm = bool(fbank_pad_before_norm)
        self.fbank_mean, self.fbank_std = float(fbank_mean), float(fbank_std)

        self.variant = variant
        self.apply_loudness_mask, self.smooth, self.overlap = bool(apply_loudness_mask), bool(smooth), overlap
        self.mix_prob, self.fit_mode, self.vanilla_alpha = float(mix_prob), fit_mode, float(vanilla_alpha)
        self.mix_stage, self.patchmix_label_mode = mix_stage, patchmix_label_mode
        self.out_len = int(round(out_len_sec * sample_rate))
        self.mix_num_threads = int(mix_num_threads)
        self.layout = ClassLayout(**(class_layout or {}))
        self.lungmix_loss = LungmixLoss(variant=variant, lambda1=lambda1, lambda2_mode=lambda2_mode, lambda2=lambda2)
        self.f_dim, self.t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)

    # ------------------------------------------------------------------ features
    def compute_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) waveform → (B, target_frames, n_mels) kaldi fbank (AudioSet constant normalization). Input is a waveform fitted to 9 s (out_len)."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        outs, n_valid = [], []
        for i in range(wav.shape[0]):
            fb = generate_fbank(wav[i : i + 1].float(), sampling_rate=self.sample_rate, n_mels=self.n_mels, normalize=False)
            n = fb.shape[0]
            if n < self.target_frames:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, self.target_frames - n))
            elif n > self.target_frames:
                fb = fb[: self.target_frames]
            outs.append(fb)
            n_valid.append(min(n, self.target_frames))
        fb = torch.stack(outs, 0)
        fb = (fb - self.fbank_mean) / (self.fbank_std * 2)
        if not self.fbank_pad_before_norm:
            # zero-pad after normalization (toolkit MelBankAudioProcessor order) — padded frames become 0
            for i, n in enumerate(n_valid):
                fb[i, n:] = 0.0
        return fb

    def forward_tokens(self, x: torch.Tensor, y=None, patch_mix: bool = False):
        """Same path as the parent forward_features, minus pooling; returns tokens (B, 2+N, 768)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # the parent hard-codes stride 10 — here the fstride/tstride flags are used (matching the patch_embed stride)
        h_patch = int((x.size()[2] - 16) / self.fstride) + 1
        w_patch = int((x.size()[3] - 16) / self.tstride) + 1
        B = x.shape[0]
        x = self.v.patch_embed(x)
        mix = None
        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=self.time_domain, hw_num_patch=[h_patch, w_patch])
            mix = (y_a, y_b, lam, index)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return (x, mix) if patch_mix else x

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean_all":
            return tokens.mean(dim=1)
        if self.pool == "mean_patches":
            return tokens[:, 2:].mean(dim=1)
        return (tokens[:, 0] + tokens[:, 1]) / 2

    def forward_features(self, x, y=None, patch_mix=False, time_domain=False):  # noqa: D401 — keeps the parent signature
        out = self.forward_tokens(x, y=y, patch_mix=patch_mix)
        if patch_mix:
            tokens, (y_a, y_b, lam, index) = out
            return self.pool_tokens(tokens), y_a, y_b, lam, index
        return self.pool_tokens(out)

    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln(self.bn(feats)))

    def apply_specaugment(self, fb: torch.Tensor) -> torch.Tensor:
        """Per-sample SpecAugment on the (B, T, F) fbank (training only, when enabled)."""
        if self.specaugment is None or not self.training:
            return fb
        outs = [self.specaugment(fb[i].transpose(0, 1)).transpose(0, 1) for i in range(fb.shape[0])]
        return torch.stack(outs, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) waveform → logits. A (B, 1024, 128) fbank passed directly is used as-is."""
        if x.dim() == 2 and x.shape[-1] != self.n_mels:
            x = self.compute_fbank(x)
        return self.classify(self.forward_features(x))

    # ------------------------------------------------------------------ training
    def _mix_waveforms(self, audio_batch: torch.Tensor, labels: torch.Tensor, valid_len=None):
        """In-batch pair mixing (CPU, global RNG — same random sequence as the official function). Returns None when no mixing is applied."""
        if self.variant == "none":
            return None
        if self.mix_prob < 1.0 and float(torch.rand(())) >= self.mix_prob:
            return None
        wav_cpu, y_cpu = audio_batch.detach().float().cpu(), labels.detach().cpu()
        vl_cpu = None
        if self.mix_stage == "raw":
            if valid_len is None:
                raise ValueError("mix_stage='raw' requires the original segment length in the batch meta_labels "
                                 "(data.params.lungmix.emit_valid_len=true)")
            vl_cpu = valid_len.detach().cpu()
        # CPU mixing is element-wise, so a single thread is fastest — with the default thread count (64) under heavy load (load 270+) a batch
        # takes 5–7 s, with 1 thread 30 ms (OMP contention). Does not affect the numerics (set via mix_num_threads; 0 leaves it untouched).

        prev_threads = torch.get_num_threads()
        if self.mix_num_threads > 0:
            torch.set_num_threads(self.mix_num_threads)
        try:
            if self.variant == "vanilla_mixup":
                mb = vanilla_mixup_batch(wav_cpu, y_cpu, alpha=self.vanilla_alpha)
            else:
                mb = lungmix_batch(wav_cpu, y_cpu, out_len=self.out_len, smooth=self.smooth,
                                   apply_loudness_mask=self.apply_loudness_mask, layout=self.layout, fit_mode=self.fit_mode,
                                   valid_len=vl_cpu, overlap=self.overlap)
        finally:
            if self.mix_num_threads > 0:
                torch.set_num_threads(prev_threads)
        dev = audio_batch.device
        mb.wav, mb.y_or, mb.y_a, mb.y_b, mb.lam, mb.index = (
            mb.wav.to(dev), mb.y_or.to(dev), mb.y_a.to(dev), mb.y_b.to(dev), mb.lam.to(dev), mb.index.to(dev))
        return mb

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(1)
        mb = self._mix_waveforms(audio_batch, labels, valid_len=meta_labels) if self.training else None
        if mb is None:
            logits = self.classify(self.forward_features(self.apply_specaugment(self.compute_fbank(audio_batch))))
            return logits, loss_func(logits, labels)

        fb = self.apply_specaugment(self.compute_fbank(mb.wav))
        if self.variant == "lungmix_patchmix":
            from rsc_toolkit.methods.patchmix import PatchMixLoss

            feats, y_a, y_b, lam, index = self.forward_features(fb, y=mb.y_or, patch_mix=True)
            logits = self.classify(feats)
            if self.patchmix_label_mode == "preserve":
                return logits, loss_func(logits, mb.y_or)
            return logits, PatchMixLoss(criterion=loss_func)(logits, y_a, y_b, lam)

        logits = self.classify(self.forward_features(fb))
        loss = self.lungmix_loss(logits, mb.y_or, mb.y_a, mb.y_b, mb.lam, criterion=loss_func)
        return logits, loss
