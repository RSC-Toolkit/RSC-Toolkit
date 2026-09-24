# Portions of this file are ported from ICASSP-2025-Lungmix (https://github.com/Tigerdwgth/ICASSP-2025-Lungmix), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Lungmix (ICASSP 2025, Ge et al., arXiv 2501.00064) waveform mixing augmentation.

Sources:
- Paper: Lungmix: A Mixup-Based Strategy for Generalization in Respiratory Sound Classification
- Official code: https://github.com/Tigerdwgth/ICASSP-2025-Lungmix lungmix.py (88 lines, no license).
  The repository contains only the augmentation function; there is no training, data, model or loss code.

Two paths side by side:

(1) `lungmix_official()` — **golden**: exact port of the official `lungmix()`, down to the order of numpy/torch global RNG calls.
    With the same seed, wav/label/lam/mask/max_length are bit-identical (scripts/reproduction/dump_lungmix_goldens.py,
    tests/unit/models/test_ast_lungmix.py). The quirks of the official code are kept too:
      - wav2 is rolled uniformly in [0, max_len), then prefixed with a **constant mean(wav2) padding** of length shift2∈[0, len(wav1)) → after re-repeating,
        the output length is max_len+shift2 (differs from 'randomly generated padding' in the Fig. 2 caption).
      - λ~Beta(0.5, 0.5) (not specified in the paper).
      - The random mask is i.i.d. Bernoulli(0.5) **per sample** at 16 kHz.
      - The loudness mask (OR of |x|>|mean+2std|), λ and the final mask {0,λ,1} are computed, but **the output waveform is blended with the random mask only**
        (lungmix.py:84-85), inconsistent with the M_Lungmix blending of paper Eq. 3. mask/lam are only returned.
      - The smooth option (max_pool1d k=201) is not in the paper (default False).
      - Label OR is string-based ('normal'/'both').

(2) `lungmix_pair(..., apply_loudness_mask=True)` — **re-implementation of paper Eq. 3 (flag)**: builds the same mask with the same RNG order, then
    actually blends x̃ = M ⊙ x_i + (1−M) ⊙ x_j, M∈{0,λ,1}. The default False gives the same output as the code version (official-code parity).
    Which of the two is the Table I 'non-linear' row is not confirmed by the authors.

Label OR is generalized to 4-class indices (`or_label`; Label Powerset, paper §II-C): normal is the identity, crackle|wheeze=both.

Mixing input length (assumption, not confirmed by the authors): the paper does not state the order of '9 s pad/cut' and mixing, but
right after Eq. 3 it writes the mask dimension as M_Lungmix ∈ {0,λ,1}^{1×maxlen(x_i,x_j)}, which presumes the two inputs have **different** lengths,
and the first __repeat_audio_to_max_length call of the official function (lungmix.py:62) is the identity on inputs already at 9 s, so both the paper and the code
support 'mix the original segments (different lengths), then fit to 9 s'.
- mix_stage='raw' (**default, main run**): cut each input to its original segment length (valid_len), mix, then repeat/cut to 9 s. The lungmix_batch(valid_len=...) path
  (the dataset passes the pre-padding length in the batch meta_labels).
- mix_stage='padded' (alternative): mix the waveforms already repeat-padded to 9 s by the dataset. Given the structure of the official function, a constant mean(wav2)
  padding of shift2∈[0, 9 s) is added, so after cut_first an expected 25% of the output is constant (differs from Fig. 2 'randomly generated padding').
  Not confirmed by the authors, so both are flags.

Mask overlap (assumption, not confirmed by the authors): paper Eq. 8 is M_Lungmix = M_Loudness ⊕ R (element-wise OR), but the official code
(lungmix.py:82-83) sets positions where random and loudness overlap to (1+λ)−1 = λ (1 under the OR reading). In the apply_loudness_mask=True path,
`overlap='lam'` (default) uses the official returned mask as-is, and `overlap='one'` uses the OR reading (random positions = 1, loudness only = λ).
Due to float32 rounding, λ appears in the official mask as two values (λ, (1+λ)−1; difference ≤ 1e-7); noted only.

Pairing (assumption, **no flag, fixed**): in-batch torch.randperm pairs; the mixed sample replaces the original (not added). Only mix_prob is a flag.
vanilla_mixup_batch uses a single λ shared by the batch (assumption).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

# ----------------------------------------------------------------------------- official code as-is (official-code parity)


def _get_aug_label(a, b):
    """Official __get__aug__label as-is (string 4-class only)."""
    if a == "normal":
        return b
    elif b == "normal":
        return a
    elif a != b:
        return "both"
    else:
        return a


def _repeat_audio_to_max_length(wav1, wav2):
    """Official __repeat_audio_to_max_length as-is."""
    max_length = max(wav1.shape[1], wav2.shape[1])
    combined_audio = torch.FloatTensor(torch.zeros(1, max_length))
    wav1 = wav1.repeat(1, math.ceil(max_length / wav1.shape[1]))[:, :max_length]
    wav2 = wav2.repeat(1, math.ceil(max_length / wav2.shape[1]))[:, :max_length]
    return wav1, wav2, max_length, combined_audio


def lungmix_official(wav1, wav2, label1, label2, smooth=False):
    """
    Bit-identical port of the official lungmix(wav1, wav2, label1, label2, smooth) (global np.random / torch RNG, same call order).
    Returns (wav, label, lam, label1, label2, wav1, wav2, mask, max_length), the same 9-tuple as the official code.
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


# ----------------------------------------------------------------------------- label OR (generalized to indices)


@dataclass(frozen=True)
class ClassLayout:
    """4-class index layout. Toolkit ICBHI default: normal 0, crackle 1, wheeze 2, both 3."""

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
    Paper Eq. 4 / §II-C Label Powerset: OR of the abnormal bits (crackle, wheeze). normal is the identity.
    Same table as the official __get__aug__label on 4 classes (normal+X→X, X+X→X, crackle+wheeze→both, both+X→both).
    """
    return layout.from_bits(layout.to_bits(y_a) | layout.to_bits(y_b))


def or_labels(y_a: torch.Tensor, y_b: torch.Tensor, layout: ClassLayout = DEFAULT_LAYOUT) -> torch.Tensor:
    out = [or_label(int(a), int(b), layout) for a, b in zip(y_a.tolist(), y_b.tolist())]
    return torch.as_tensor(out, dtype=torch.long, device=y_a.device)


# ----------------------------------------------------------------------------- waveform mixing (code version / paper version)


@dataclass
class LungmixResult:
    wav: torch.Tensor            # (1, L_out), L_out = max_len + shift2 (official quirk; cut by the caller)
    lam: float
    mask: torch.Tensor           # (1, L_out) float ∈ {0, λ, 1}, same as the official returned mask
    random_mask: torch.Tensor    # (1, L_out) bool
    loudness_mask: torch.Tensor  # (1, L_out) float (before multiplying by λ, smooth applied)
    wav1: torch.Tensor           # wav1 after processing (1, L_out)
    wav2: torch.Tensor           # wav2 after roll, padding, re-repeat (1, L_out)
    max_length: int
    blend_mask: Optional[torch.Tensor] = None  # mask actually used for Eq. 3 blending (with apply_loudness_mask=False, the official mask, unrelated to random_mask)


def lungmix_pair(
    wav1: torch.Tensor,
    wav2: torch.Tensor,
    smooth: bool = False,
    apply_loudness_mask: bool = False,
    overlap: str = "lam",
) -> LungmixResult:
    """
    Mixes a pair of waveforms with Lungmix. RNG call order and mask computation are the same as lungmix_official.

    apply_loudness_mask=False (default, official-code parity): as in the official code, output = random_mask*wav1 + (~random_mask)*wav2.
    apply_loudness_mask=True (re-implementation of paper Eq. 3): output = M*wav1 + (1-M)*wav2, M∈{0,λ,1}.
      overlap='lam' (default): M = official returned mask (random∧loudness overlap = (1+λ)−1 = λ).
      overlap='one'          : M = Eq. 8 OR reading (random positions = 1, loudness only = λ, else 0).
    The returned mask is always the official returned mask (for golden comparison); the mask actually used for blending is blend_mask.
    """
    if overlap not in OVERLAP_MODES:
        raise ValueError(f"overlap must be one of {OVERLAP_MODES}: {overlap!r}")
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
    mask = mask - (mask > 1).float()          # {0, λ, 1}: random only → 1, loudness only → λ, both → λ(1+λ−1)
    blend_mask = mask.float()
    if apply_loudness_mask:
        if overlap == "one":                  # Eq. 8 OR reading: random positions are 1
            blend_mask = torch.where(random_mask, torch.ones_like(loudness_lam), loudness_lam).float()
        wav = blend_mask * wav1 + (1.0 - blend_mask) * wav2            # paper Eq. 3
    else:
        combined_audio[:, : wav1.shape[1]] += wav1[:, :]
        wav = (random_mask) * combined_audio + (~random_mask) * wav2[:, :]   # official code lines 84-85
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
    Fits the mixing output (length max_len+shift2) to the training length. Repeat-pads if shorter.
    Assumption (not confirmed by the authors): default cut_first (first out_len). Alternatives: cut_last (last out_len; discards the most
    shift2 constant padding), random_crop (random start, consumes np.random), none (unchanged).
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
    y_or: torch.Tensor     # (B,) OR label
    y_a: torch.Tensor      # (B,) label of the first input
    y_b: torch.Tensor      # (B,) label of the partner
    lam: torch.Tensor      # (B,) float, λ differs per pair
    index: torch.Tensor    # (B,) partner index


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
    Feeds a batch (B, L) sample by sample to lungmix_pair (mixes i with index[i]; result = same as the per-sample function).
    If index is None, torch.randperm(B) (assumption, no flag: in-batch permutation pairs replacing the original;
    the probability is the caller's mix_prob).
    If valid_len (B,) is given, only the first valid_len[i] samples of each waveform (original segment, before padding) are mixed (mix_stage='raw', default, main run);
    if None, the whole (B, L) (already padded waveforms, mix_stage='padded' alternative).
    The global RNGs (np.random / torch) are used as-is, so call with CPU tensors to get the same random sequence as the official function.
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
            raise ValueError(f"valid_len length {len(vl)} != batch {B}")
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
    Table I 'vanilla Mixup' baseline: linear waveform mixup x̃ = λ x_i + (1−λ) x_j, one λ~Beta(α,α) for the batch.
    Assumptions (not specified in the paper): α=0.5 (matching Beta(0.5,0.5) of the official lungmix), mixing at the waveform stage. y_or is the OR label (for comparison).
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
