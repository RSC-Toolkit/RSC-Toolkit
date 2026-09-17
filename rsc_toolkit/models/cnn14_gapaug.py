"""
GaP-Aug (Chang et al., ICASSP 2024) classifier — original PANNs Cnn14 + the training procedure of paper Algorithm 1. **Re-implementation from the paper description** (no official or unofficial code).

Backbone (same structure as Cnn14 in the official PANNs code qiuqiangkong/audioset_tagging_cnn pytorch/models.py — verified by golden comparison):
  waveform → power STFT(window, center, reflect) → mel(librosa slaney) → 10·log10(clamp amin 1e-10, ref 1, top_db None)
  → bn0(BatchNorm2d(64), mel axis) → conv_block1–6(each 2×(3×3 conv+BN+ReLU), avg pool 2×2 / last 1×1, dropout 0.2)
  → mean over frequency (mel) axis (mean dim=3) → max+mean over time [= pooled 2048] → dropout 0.5 → fc1(2048→2048)+ReLU [= feat] → dropout 0.5 [= emb, PANNs 'embedding']
  → classification fc(2048→n_cls) — its input is chosen by the head_input (emb|feat) flag.
  ConvBlock is reused from the toolkit's models/backbone/cnn.py (same as official: 3×3 ×2, no bias, xavier/BN 1·0 init).
  Pretraining: AudioSet Cnn14 16 kHz version (Cnn14_16k_mAP=0.438.pth, zenodo 3987831; the sister paper JMIR AI 2025 uses the 16 kHz version) —
  all keys except spectrogram/logmel/fc_audioset are loaded strictly. The 32 kHz version (zenodo 3960586) is the pretrained="32k" flag
  (trained at 32 kHz / fmax 14 kHz, so sample_rate/fmax/data config must change together — a warning is printed).

Features (paper §3.1 '64-dim log Mel filterbank, 25 ms / 6.25 ms'): 16 kHz, 10 s, 64 mel → n_fft 400 / hop 100 (feature_preset="paper").
  The official 16k ckpt was trained with 512/160 (feature_preset="panns16k"; golden comparison uses this setting).
  **Feature definition = PANNs style (assumption)**: power spectrum → slaney mel (fmin 50 / fmax 8000) → 10·log10 dB (ref 1, amin 1e-10, top_db None).
  The paper wording could also be read as a kaldi fbank (HTK mel, natural log), but the backbone is PANNs, so the PANNs front-end definition is used.
  Window function: the official ckpt was trained with hann → default hann; the sister paper (JMIR AI 2025) states
  'Hamming window 512 / hop 160' for the same lab's CNN14 pipeline → window="hamming" flag (assumption).

Assumptions (not confirmed by the authors; all are config flags):
  norm_mode        : log-mel made non-negative = per-sample min-max [0,1] (the model input uses that scale; same for training and evaluation). Alternatives batch_minmax / db_offset / none /
                     sample_minmax_restore (γ applied only in [0,1], model input restored to the original dB scale — an open question in the paper description).
                     Side effect: the pretrained bn0 running stats assume PANNs dB input (per-mel mean −46..−18, std ~20), so with [0,1] input
                     early eval outputs are nearly constant, and bn0 running stats re-adapt during training (momentum 0.1, ~tens of steps).
  mixup_alpha      : 1.0 (rationale: official PANNs pipeline main.py:174 Mixup(mixup_alpha=1.) + do_mixup even/odd pairs = structure of Algo. 1 lines 7-12;
                     but PANNs mixes **after** bn0, whereas here the spectrogram before bn0 is mixed) — still an assumption,  mixup_loss: soft (alternative pair)
  triplet          : **w_trip 1.0** = unweighted sum 'L ← L_NLL + L_triplet' of paper Algorithm 1 line 14 (literal). 0.01 is an **intentional deviation flag**
                     (w_trip=0.01) based on the sister paper (JMIR AI 2025, CE-based, different setting, λ=0.01).
                     margin 1.0, X̃_a samples (triplet_source), random triplets,
                     **triplet_layer** = emb (PANNs 'embedding' = fc1 ReLU then dropout 0.5; the dropout mask applies in training mode)
                     | feat (fc1 ReLU, before dropout) | pooled (conv pooling 2048, before fc1) — the layer is not specified in the paper
  head_input       : **emb** (official PANNs finetune_template.py Transfer_Cnn14 = fc_transfer on the post-dropout embedding → log_softmax → NLL;
                     consistent with the 'NLL loss' wording of paper §2.3) | feat (directly on the fc1 ReLU output, no dropout) — not specified
  joint_forward    : **False (default, literal)** = two forwards M(X̃_mix)·M(X̃_a) as in Algo. 1 line 13 (separate BN batch statistics).
                     True packs all 16 into one batch for a single forward (shared BN statistics; speed optimization, not literal)
  head_init        : classification fc xavier_uniform (bias 0) (PANNs init_layer convention)
  window           : hann (ckpt training setting) | hamming (sister paper convention)
  data-side assumptions: cycles longer than 10 s are truncated at the front (pad_type repeat; the paper's 'divided into 10-second segments' could
                     also mean splitting into multiple segments), no per-recording fade (configs/data/icbhi_gapaug.yaml `fade: false`; not mentioned in the paper)
  specaug/patchmask: see augmentations/gap_aug.py

method → (augmentation, mixup): gapaug(gamma, on) | gapaug_nomixup(gamma, off) | naive(none, off) | specaug(spec, on) |
  patchmask(mask, on) | mixup_only(none, on). The `mixup` argument overrides it individually (specaug/patchmask w/o Mixup).
The loss is the same for every method (NLL + w_trip·triplet, default w_trip 1; Table 2 compares augmentations, so the loss is fixed — assumption). Evaluation: loss_func(CE).
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from rsc_toolkit.augmentations.gap_aug import GammaPatchAug, SpecAugPANNs, normalize_nonneg, sample_minmax_restore, sample_minmax_split
from rsc_toolkit.methods.gapaug import mixup_pairs, pair_nll, soft_target_nll, split_odd_even, triplet_loss
from rsc_toolkit.models.backbone.cnn import ConvBlock, _init_bn, _init_layer

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)

METHODS: Dict[str, tuple] = {
    "gapaug": ("gamma", True),
    "gapaug_nomixup": ("gamma", False),
    "naive": ("none", False),
    "specaug": ("spec", True),
    "patchmask": ("mask", True),
    "mixup_only": ("none", True),
}

PANNS_CNN14 = {
    "16k": {
        "url": "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
        "path": "pretrained_models/upstream/Cnn14_16k_mAP=0.438.pth",
        "sha256": "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6",  # verified (358,668,570 B)
        "size": 358_668_570,
        "sample_rate": 16000, "fmax": 8000.0,
    },
    "32k": {
        "url": "https://zenodo.org/record/3960586/files/Cnn14_mAP%3D0.431.pth?download=1",
        "path": "pretrained_models/upstream/Cnn14_mAP=0.431.pth",
        "sha256": None,  # not downloaded yet — recorded in provenance.json on first download
        "size": None,
        "sample_rate": 32000, "fmax": 14000.0,
    },
}
PANNS_LICENSE = "MIT (PANNs, qiuqiangkong/audioset_tagging_cnn) — not bundled or redistributed with the package, downloaded separately"
FEATURE_PRESETS = {
    "paper": dict(n_fft=400, hop=100),      # 25 ms / 6.25 ms @16 kHz (paper §3.1)
    "panns16k": dict(n_fft=512, hop=160),   # official 16k ckpt training setting (README) — golden comparison
}
WINDOWS = {"hann": torch.hann_window, "hamming": torch.hamming_window}
EXCLUDE_PREFIXES = ("spectrogram_extractor.", "logmel_extractor.", "fc_audioset.")
HEAD_INPUTS = ("emb", "feat")
TRIPLET_LAYERS = ("emb", "feat", "pooled")
NORM_MODES = ("sample_minmax", "sample_minmax_restore", "batch_minmax", "db_offset", "none")


class PANNsLogMel(nn.Module):
    """torchlibrosa Spectrogram(power) + LogmelFilterBank(librosa slaney mel, log10, ref 1, amin 1e-10, top_db None) reproduced with torchaudio."""

    def __init__(self, sample_rate=16000, n_fft=512, hop=160, n_mels=64, fmin=50.0, fmax=8000.0, ref=1.0, amin=1e-10, window="hann"):
        super().__init__()
        if window not in WINDOWS:
            raise ValueError(f"window must be one of {tuple(WINDOWS)}, got {window!r}")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop, f_min=fmin, f_max=fmax, n_mels=n_mels,
            window_fn=WINDOWS[window], power=2.0, center=True, pad_mode="reflect", norm="slaney", mel_scale="slaney",
        )
        self.window = window
        self.ref, self.amin = float(ref), float(amin)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, 1, T, n_mels) — PANNs layout."""
        m = self.mel(wav)                                   # (B, n_mels, T)
        db = 10.0 * torch.log10(torch.clamp(m, min=self.amin)) - 10.0 * torch.log10(torch.tensor(max(self.amin, self.ref)))
        return db.transpose(1, 2).unsqueeze(1).contiguous()


class Cnn14Backbone(nn.Module):
    """The part of PANNs Cnn14 after bn0 (input (B,1,T,F) log-mel).
    Returns dict(pooled=conv pooling 2048 (before fc1 and dropout), feat=fc1 ReLU output (before dropout), emb=after dropout = PANNs 'embedding')."""

    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        _init_bn(self.bn0)
        _init_layer(self.fc1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for i, blk in enumerate((self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6)):
            x = blk(x, pool_size=(1, 1) if i == 5 else (2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        pooled = x1 + x2
        x = F.dropout(pooled, p=0.5, training=self.training)
        feat = F.relu_(self.fc1(x))
        emb = F.dropout(feat, p=0.5, training=self.training)
        return {"pooled": pooled, "feat": feat, "emb": emb}


@ModelFactory.register_model_class("cnn14_gapaug")
class Cnn14GaPAug(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        method: str = "gapaug",
        mixup: Optional[bool] = None,          # None = method default; True/False overrides (SpecAug/PatchMask w/o Mixup)
        pretrained: Optional[str] = "16k",     # 16k | 32k | null
        pretrained_path: Optional[str] = None, # explicit path
        # ---- features ----
        sample_rate: int = 16000,
        feature_preset: str = "paper",         # paper(400/100) | panns16k(512/160)
        n_fft: Optional[int] = None,
        hop: Optional[int] = None,
        n_mels: int = 64,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        window: str = "hann",                  # hann (ckpt training setting) | hamming (sister paper convention)  (assumption)
        norm_mode: str = "sample_minmax",      # sample_minmax | sample_minmax_restore | batch_minmax | db_offset | none  (assumption)
        db_offset: float = 100.0,
        # ---- GaP-Aug / PatchMask ----
        n_patches: int = 32,
        max_freq: int = 16,
        max_time: int = 16,
        min_freq: int = 1,
        min_time: int = 1,
        gamma_min: float = 1.7,
        gamma_max: float = 2.0,
        gain: float = 1.0,
        gamma_per: str = "patch",
        aug_p: float = 1.0,
        # ---- SpecAug baseline (PANNs defaults assumed) ----
        specaug_time_width: int = 64,
        specaug_time_stripes: int = 2,
        specaug_freq_width: int = 8,
        specaug_freq_stripes: int = 2,
        # ---- mixup / loss ----
        mixup_alpha: float = 1.0,              # assumption
        mixup_loss: str = "soft",              # soft | pair
        w_trip: float = 1.0,                   # Algo. 1 line 14 unweighted sum (literal). 0.01 = deviation flag based on the sister paper
        triplet_margin: float = 1.0,
        triplet_p: float = 2.0,
        triplet_source: str = "xa",            # xa (Algo. 1) | all (whole unmixed batch; alternative)
        triplet_layer: str = "emb",            # emb | feat | pooled  (assumption: embedding layer not specified)
        head_input: str = "emb",               # emb (official Transfer_Cnn14 convention) | feat  (assumption: not specified)
        joint_forward: bool = False,           # False = Algo. 1 line 13 literal (2 forwards) | True = single joint forward (not literal)
        head_init: str = "xavier",             # xavier | default
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {tuple(METHODS)}, got {method!r}")
        if feature_preset not in FEATURE_PRESETS:
            raise ValueError(f"feature_preset must be one of {tuple(FEATURE_PRESETS)}")
        if mixup_loss not in ("soft", "pair"):
            raise ValueError("mixup_loss must be soft|pair")
        if triplet_source not in ("xa", "all"):
            raise ValueError("triplet_source must be xa|all")
        if triplet_layer not in TRIPLET_LAYERS:
            raise ValueError(f"triplet_layer must be one of {TRIPLET_LAYERS}, got {triplet_layer!r}")
        if head_input not in HEAD_INPUTS:
            raise ValueError(f"head_input must be one of {HEAD_INPUTS}, got {head_input!r}")
        self.method = method
        self.aug_kind, default_mix = METHODS[method]
        self.use_mixup = default_mix if mixup is None else bool(mixup)
        self.n_cls = int(n_cls)
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")
        self.norm_mode, self.db_offset = norm_mode, float(db_offset)
        self.mixup_alpha, self.mixup_loss = float(mixup_alpha), mixup_loss
        self.w_trip, self.triplet_margin, self.triplet_p, self.triplet_source = float(w_trip), float(triplet_margin), float(triplet_p), triplet_source
        self.triplet_layer, self.head_input = triplet_layer, head_input
        self.joint_forward = bool(joint_forward)
        self.sample_rate, self.fmax, self.window = int(sample_rate), float(fmax), window

        preset = FEATURE_PRESETS[feature_preset]
        self.n_fft, self.hop = int(n_fft or preset["n_fft"]), int(hop or preset["hop"])
        self.frontend = PANNsLogMel(sample_rate, self.n_fft, self.hop, n_mels, fmin, fmax, window=window)
        self.backbone = Cnn14Backbone(n_mels)
        self.fc = nn.Linear(2048, self.n_cls)
        if head_init == "xavier":
            _init_layer(self.fc)

        self.gap = GammaPatchAug(n_patches, max_freq, max_time, min_freq, min_time, (gamma_min, gamma_max), gain, gamma_per, aug_p, mode="gamma")
        self.patchmask = GammaPatchAug(n_patches, max_freq, max_time, min_freq, min_time, (gamma_min, gamma_max), gain, gamma_per, aug_p, mode="mask")
        self.specaug = SpecAugPANNs(specaug_time_width, specaug_time_stripes, specaug_freq_width, specaug_freq_stripes, aug_p)

        self.pretrained_info = None
        if pretrained_path is not None or pretrained:
            self.load_panns_weights(pretrained_path, pretrained or "16k")
        logger.info("Cnn14GaPAug(method=%s, aug=%s, mixup=%s, feat=%d/%d %s, norm=%s, w_trip=%.3g@%s, alpha=%.2f, head=%s)",
                    method, self.aug_kind, self.use_mixup, self.n_fft, self.hop, window, norm_mode, self.w_trip, triplet_layer,
                    self.mixup_alpha, head_input)

    # ------------------------------------------------------------------ pretraining
    def load_panns_weights(self, path: Optional[str], variant: str = "16k") -> Dict[str, list]:
        from rsc_toolkit.paths import download_target, resolve_pretrained

        if path is None:
            spec = PANNS_CNN14[variant]
            if spec["sample_rate"] != self.sample_rate or spec["fmax"] != self.fmax:
                logger.warning("PANNs Cnn14 %s ckpt was trained at %d Hz / fmax %.0f but the model uses %d Hz / fmax %.0f — "
                               "align the data config (sampling_rate) and fmax as well",
                               variant, spec["sample_rate"], spec["fmax"], self.sample_rate, self.fmax)

            ckpt = resolve_pretrained([spec["path"]]) or download_target(spec["path"])  # RSCTK_PRETRAINED_ROOT, then cwd
            if not ckpt.exists():
                from rsc_toolkit.datasets.download_utils import download_file, record_download_provenance

                ckpt.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading PANNs Cnn14 %s: %s", variant, spec["url"])
                download_file(spec["url"], ckpt)
                record_download_provenance(ckpt, url=spec["url"], known_size=spec["size"], known_sha256=spec["sha256"])
                self._append_provenance_extras(ckpt, variant)
        else:
            ckpt = resolve_pretrained(explicit=path)   # relative path: cwd, then under RSCTK_PRETRAINED_ROOT
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        kept = {k: v for k, v in state.items() if not k.startswith(EXCLUDE_PREFIXES)}
        self.backbone.load_state_dict(kept, strict=True)
        self.pretrained_info = {"path": str(ckpt), "variant": variant, "n_loaded": len(kept),
                                "excluded": sorted(k for k in state if k.startswith(EXCLUDE_PREFIXES))}
        logger.info("Loaded PANNs Cnn14: %s (%d keys, %d excluded)", ckpt, len(kept), len(self.pretrained_info["excluded"]))
        return self.pretrained_info

    @staticmethod
    def _append_provenance_extras(ckpt: pathlib.Path, variant: str) -> None:
        """Merges license/usage fields that record_download_provenance does not write into provenance.json (kept across re-downloads)."""
        p = ckpt.with_name(ckpt.name + ".provenance.json")
        try:
            rec = json.loads(p.read_text()) if p.exists() else {}
            rec.update({"license": PANNS_LICENSE, "used_by": f"rsc_toolkit/models/cnn14_gapaug.py (GaP-Aug, {variant} version)"})
            p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        except OSError as e:  # a provenance write failure does not block training
            logger.warning("Failed to write extra provenance fields %s: %s", p, e)

    # ------------------------------------------------------------------ features
    def _as_wave(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"expected waveform (B, L), got {tuple(x.shape)}")
        return x

    def compute_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform (B,L) → PANNs log-mel (B,1,T,F) (before normalization; for golden comparison)."""
        return self.frontend(self._as_wave(wav))

    def make_input(self, wav: torch.Tensor) -> torch.Tensor:
        """log-mel → non-negative (norm_mode) — input scale shared by training and evaluation (with sample_minmax_restore the model input is on the original dB scale)."""
        with torch.no_grad():
            return normalize_nonneg(self.compute_logmel(wav), self.norm_mode, self.db_offset)

    def make_augmented_input(self, wav: torch.Tensor) -> torch.Tensor:
        """Training path: non-negative → augmentation (Algo. 1 lines 5-6). With sample_minmax_restore, γ is applied in [0,1], then restored to the dB scale."""
        with torch.no_grad():
            lm = self.compute_logmel(wav)
            if self.norm_mode == "sample_minmax_restore":
                x01, mn, mx = sample_minmax_split(lm)
                return sample_minmax_restore(self.augment(x01), mn, mx)
            return self.augment(normalize_nonneg(lm, self.norm_mode, self.db_offset))

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.aug_kind == "gamma":
            return self.gap(x)
        if self.aug_kind == "mask":
            return self.patchmask(x)
        if self.aug_kind == "spec":
            return self.specaug(x)
        return x

    # ------------------------------------------------------------------ forward
    def forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(x)
        out["logits"] = self.fc(out[self.head_input])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_backbone(self.make_input(x))["logits"]

    def _trip(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return out[self.triplet_layer]

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if not self.training:
            logits = self.forward(audio_batch)
            return logits, loss_func(logits, labels)

        x = self.make_augmented_input(audio_batch)  # (B,1,T,F): non-negative + per-sample augmentation of Algo. 1 lines 5-6
        B = x.shape[0]

        if not self.use_mixup or B < 2:
            out = self.forward_backbone(x)
            loss = F.cross_entropy(out["logits"], labels)  # NLL(one-hot) == CE
            if self.w_trip > 0:
                loss = loss + self.w_trip * triplet_loss(self._trip(out), labels, self.triplet_margin, self.triplet_p)
            return out["logits"], loss

        idx_a, idx_b = split_odd_even(B, device=x.device)
        xa, xb, ya, yb = x[idx_a], x[idx_b], labels[idx_a], labels[idx_b]
        x_mix, y_mix, lam = mixup_pairs(xa, xb, ya, yb, self.n_cls, self.mixup_alpha)

        if self.joint_forward:
            out = self.forward_backbone(torch.cat([x_mix, xa], dim=0))
            n = x_mix.shape[0]
            logits_mix, logits_a, emb_a = out["logits"][:n], out["logits"][n:], self._trip(out)[n:]
        else:
            om = self.forward_backbone(x_mix)
            oa = self.forward_backbone(xa)
            logits_mix, logits_a, emb_a = om["logits"], oa["logits"], self._trip(oa)

        if self.mixup_loss == "soft":
            loss = soft_target_nll(logits_mix, y_mix)
        else:
            loss = pair_nll(logits_mix, ya, yb, lam)
        if self.w_trip > 0:
            if self.triplet_source == "xa":
                loss = loss + self.w_trip * triplet_loss(emb_a, ya, self.triplet_margin, self.triplet_p)
            else:
                ob = self.forward_backbone(xb)
                loss = loss + self.w_trip * triplet_loss(torch.cat([emb_a, self._trip(ob)]), torch.cat([ya, yb]), self.triplet_margin, self.triplet_p)

        # Logits for training metrics (B, n_cls): a positions = M(X̃_a) (valid against hard labels), b positions = M(X_mix) (mixed input, unrelated to b labels).
        # → train/topk_accuracy during training is not meaningful (half are mixed predictions); only the loss is. Evaluation metrics use the eval path (forward).
        full = x.new_zeros((B, self.n_cls))
        full[idx_a] = logits_a.detach().to(full.dtype)
        full[idx_b] = logits_mix.detach().to(full.dtype)
        if B % 2 == 1:
            full[B - 1] = logits_a.detach().mean(0)
        return full, loss

    # ------------------------------------------------------------------ diagnostics
    def describe(self) -> dict:
        n_all = sum(p.numel() for p in self.parameters())
        return {"params_total": n_all, "params_backbone": sum(p.numel() for p in self.backbone.parameters()),
                "method": self.method, "aug": self.aug_kind, "mixup": self.use_mixup, "n_fft": self.n_fft, "hop": self.hop,
                "window": self.window, "norm_mode": self.norm_mode, "head_input": self.head_input,
                "triplet_layer": self.triplet_layer, "pretrained": self.pretrained_info}
