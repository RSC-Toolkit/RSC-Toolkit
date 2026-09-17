"""
Four audio augmentations of Domain Transfer (Wang & Wang, ICASSP 2022): re-implemented from the paper specification (no code released).

Source: "A Domain Transfer Based Data Augmentation Method for Automated Respiratory Classification",
ICASSP 2022 pp.9017–9020, DOI 10.1109/ICASSP43922.2022.9746941 (§2.1).

Stated in the paper:
- Traditional augmentation: **noise addition (white noise + pre-recorded environmental noise, SNR randomly sampled from 1–10)**, time stretching, pitch shifting
- **Splicing**: concatenates short samples of the same class (the same number of spliced samples per class → mitigates imbalance)
- **Circular padding**: replaces zero padding and is **also applied to the test set**
- **Mixup**: linear combination of spectrograms and labels, α = 0.2

Assumptions (not specified in the paper; all exposed as flags):
- **SNR is in dB**, interpreted as uniform sampling over 1–10 dB (the paper only says "randomly sampled between 1 and 10").
- **No environmental noise (white noise only)**: the paper does not name the corpus, and no external corpus outside the toolkit's
  data scope (ICBHI, SPRSound) is pulled in.
- Scale ranges of time stretch / pitch shift not specified → common values (0.9–1.1× speed, ±2 semitones) as defaults, exposed as flags.
- Criterion and maximum count for the "short samples" to splice not specified → interpreted as concatenating cycles shorter than the target length up to the target length.
- **Resampling interpretation (`impl` flag)**: the default `linear` is a linear-interpolation resample, so length and
  pitch change together. This differs from the conventional meaning (pitch-preserving phase-vocoder stretch), and pitch_shift
  only adds length restoration on top of the same resample, so **the two augmentations become nearly the same operation with different scale ranges**;
  the independent effects of the two augmentations the paper lists separately are not reproduced under this interpretation. `impl="phase_vocoder"` (data.params.stretch_impl)
  is the conventional interpretation (pitch-preserving stretch, length-preserving pitch shift).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def add_white_noise(x: torch.Tensor, snr_db: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Adds white Gaussian noise at the target SNR (dB).

    White noise = i.i.d. normal random samples (flat power spectrum over the whole band).
    The scale is set from the signal power:
        P_x = mean(x²)                     signal power
        P_n = P_x / 10^(SNR_dB / 10)       target noise power
        n   = N(0,1) samples → rescaled to power P_n
        out = x + n
    At SNR 1 dB the noise power is about 79 % of the signal, at 10 dB 10 % (consistent with the paper's "strong noise augmentation").
    For silence (P_x = 0) the input is returned unchanged, so the output is not pure noise.
    """
    p_x = torch.mean(x.float() ** 2)
    if float(p_x) <= 0.0:
        return x
    p_n = p_x / (10.0 ** (snr_db / 10.0))
    n = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    n = n * torch.sqrt(p_n / torch.clamp(torch.mean(n.float() ** 2), min=1e-12))
    return x + n


def sample_snr_db(low: float = 1.0, high: float = 10.0, generator: Optional[torch.Generator] = None) -> float:
    """Paper §2.1 'SNR was randomly sampled between 1 and 10', interpreted as uniform sampling in dB (assumption)."""
    return float(torch.empty(1).uniform_(low, high, generator=generator).item())


def circular_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Paper §2.1 Circular Padding: fills the target length by repeating the signal cyclically instead of zero padding.
    **Also applied to the test set** (stated in the paper). Longer signals are truncated from the start (paper: 'truncate the too-long audio').
    """
    n = int(x.shape[-1])
    if n == target_len:
        return x
    if n > target_len:
        return x[..., :target_len]
    reps = -(-target_len // n)  # ceil
    return x.repeat(*([1] * (x.dim() - 1)), reps)[..., :target_len]


def splice_same_class(clips: Sequence[torch.Tensor], target_len: int) -> torch.Tensor:
    """
    Paper §2.1 Splicing Audio: concatenates short cycles of **the same class** to reach the target length.
    If still short, finishes with circular padding (the paper does not describe length handling after splicing, so it is handled consistently with circular padding).
    """
    if not clips:
        raise ValueError("splice_same_class: empty clip list")
    out = torch.cat([c.reshape(1, -1) for c in clips], dim=-1)
    return circular_pad(out, target_len)


def mixup_batch(
    x: torch.Tensor, y_onehot: torch.Tensor, alpha: float = 0.2, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paper Eq. (1): linear combination of spectrogram and one-hot label with the same λ. λ ~ Beta(α, α), α = 0.2.
    Pairs come from a random permutation within the batch (the paper says 'randomly draw another sample xb from the dataset').
    """
    if alpha <= 0:
        return x, y_onehot
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.shape[0], device=x.device, generator=generator)
    return lam * x + (1.0 - lam) * x[idx], lam * y_onehot + (1.0 - lam) * y_onehot[idx]


def _stft_params(n: int):
    """STFT settings for the phase vocoder: 64 ms window (n_fft 512) / 16 ms hop for an 8 kHz waveform. The window shrinks for short waveforms."""
    n_fft = min(512, max(32, 1 << (max(1, n).bit_length() - 1)))
    return n_fft, n_fft // 4


def time_stretch(x: torch.Tensor, rate: float, impl: str = "linear") -> torch.Tensor:
    """
    Time stretching: changes the length by factor rate (rate > 1 shortens). The paper does not give the range (assumption: 0.9–1.1).

    impl="linear"        : linear-interpolation resample; **pitch changes as well** (default).
    impl="phase_vocoder" : STFT phase vocoder; conventional time stretching that **preserves pitch**.
    """
    n = int(x.shape[-1])
    new_n = max(1, int(round(n / rate)))
    if impl == "linear":
        return torch.nn.functional.interpolate(
            x.reshape(1, 1, n).float(), size=new_n, mode="linear", align_corners=False
        ).reshape(1, new_n).to(x.dtype)
    if impl != "phase_vocoder":
        raise ValueError(f"impl must be linear|phase_vocoder, got {impl!r}")
    import torchaudio

    n_fft, hop = _stft_params(n)
    win = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x.reshape(-1).float(), n_fft, hop, window=win, return_complex=True)   # (F, T)
    phase_advance = torch.linspace(0, math.pi * hop, spec.shape[0], device=x.device)[..., None]
    stretched = torchaudio.functional.phase_vocoder(spec, float(rate), phase_advance)
    y = torch.istft(stretched, n_fft, hop, window=win, length=new_n)
    return y.reshape(1, new_n).to(x.dtype)


def pitch_shift(x: torch.Tensor, n_steps: float, sample_rate: int, impl: str = "linear") -> torch.Tensor:
    """
    Pitch shifting: shifts pitch by n_steps semitones while preserving length. The paper does not give the range (assumption: ±2 semitones).

    impl="linear"        : shifts pitch by resampling, then restores the original length (circular pad / truncation); built on the same operation as time_stretch(linear) (default).
    impl="phase_vocoder" : torchaudio.functional.pitch_shift; conventional pitch shift that stretches with a phase vocoder and resamples back.
    """
    n = int(x.shape[-1])
    if impl == "phase_vocoder":
        import torchaudio

        n_fft, hop = _stft_params(n)
        y = torchaudio.functional.pitch_shift(x.reshape(1, n).float(), int(sample_rate), float(n_steps),
                                              n_fft=n_fft, hop_length=hop)
        return y.reshape(1, n).to(x.dtype)
    if impl != "linear":
        raise ValueError(f"impl must be linear|phase_vocoder, got {impl!r}")
    factor = 2.0 ** (n_steps / 12.0)
    stretched = time_stretch(x, 1.0 / factor)  # length change (pitch changes)
    return circular_pad(stretched, n) if stretched.shape[-1] < n else stretched[..., :n]
