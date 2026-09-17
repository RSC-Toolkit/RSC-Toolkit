"""
Official-code parity golden tests for the Patch-Mix data pipeline.

Goldens were produced by running the authors' code (get_individual_cycles_torchaudio -> cut_pad_sample_torchaudio ->
generate_fbank, SpecAugment) on the same torchaudio
(scripts/reproduction/dump_patchmix_pipeline_goldens.py — the original repository does not pin the torchaudio
version, so the code itself is the reference).

Our chain pinned here:
  load_audio(fade=True)  ==  fade(sr/16, linear) over the whole recording
  slice_audio            ==  _slice_data_torchaudio
  AudioPreprocessor(pad_type="repeat_fade")  ==  cut_pad_sample_torchaudio
  processors.mel.generate_fbank              ==  authors' generate_fbank (only the axis order differs)
  augmentations.SpecAugment (freq-first)     ==  authors' SpecAugment (time-first)

The SpecAugment golden also shows that skipping about 16% via the `p >= torch.randn(1)` gate,
even at p=1.0, is **present in the original code**, and that our port produces the same output
with the same RNG consumption order.
"""
import json
import pathlib
import random

import numpy as np
import pytest
import torch

from rsc_toolkit.augmentations.specaugment import SpecAugment
from rsc_toolkit.datasets.icbhi.processing import load_audio, slice_audio
from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor
from rsc_toolkit.processors.mel import generate_fbank


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "patchmix_pipeline"
DATA = REPO / "sample_data" / "toy_dataset" / "files"

needs_golden = pytest.mark.skipif(
    not (GOLDEN / "fbank_golden.pt").exists(),
    reason="golden missing — generate with scripts/reproduction/dump_patchmix_pipeline_goldens.py",
)


@needs_golden
def test_cycle_to_fbank_chain_matches_reference():
    meta = json.load(open(GOLDEN / "cycles_meta.json"))
    golden = torch.load(GOLDEN / "fbank_golden.pt")  # (N, 798, 128)

    preprocessor = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat_fade"))

    i = 0
    for entry in meta["entries"]:
        wav_path = DATA / f"{entry['filename']}.wav"
        wav = load_audio(wav_path, target_sampling_rate=16000, fade=True)
        for cyc in entry["cycles"]:
            audio = slice_audio(wav, t0=cyc["start"], t1=cyc["end"], sampling_rate=16000)
            audio = preprocessor(audio)

            fp = cyc["audio_fingerprint"]
            assert list(audio.shape) == fp["shape"]
            # fade_out was actually applied (mean absolute value of the last 1000 samples matches too)
            assert float(audio[..., -1000:].abs().mean()) == pytest.approx(fp["tail_mean_abs"], abs=1e-6)

            ours = generate_fbank(audio, sampling_rate=16000, n_mels=128)  # (798, 128)
            ref = golden[i]
            diff = float((ours - ref).abs().max())
            assert torch.allclose(ours, ref, atol=1e-5), (
                f"{entry['filename']} cycle{i}: fbank mismatch max|diff|={diff:.3e}"
            )
            i += 1
    assert i == golden.shape[0]


@needs_golden
@pytest.mark.exact_golden  # bitwise comparison — dev machine only (different CPU/BLAS/thread counts diverge in the last digits)
def test_specaugment_matches_reference_with_same_seeds():
    """
    With the same seed, the authors' SpecAugment ([c, time, freq] input, internal transpose) and
    our SpecAugment (freq-first input) give identical results.
    This holds only if the RNG consumption order (torch.randn gate -> np.uniform/randint masks) is the same.
    """
    img = torch.load(GOLDEN / "specaug_input.pt")  # [1, 798, 128] (time-first)
    ref_out = torch.load(GOLDEN / "specaug_golden.pt")  # [1, 798, 128]

    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)

    ours_in = img.squeeze(0).transpose(0, 1).clone()  # (128, 798) freq-first
    ours_out = SpecAugment(policy="icbhi_ast_sup", mask="mean")(ours_in)

    assert torch.equal(ours_out.transpose(0, 1), ref_out.squeeze(0))
    # masking actually happened (dumped with a seed that passes the gate)
    assert not torch.equal(ours_out, ours_in)


def test_repeat_fade_applies_fade_even_when_length_matches():
    """
    Quirk of the original cut_pad_sample_torchaudio: even when the length exactly equals the target,
    it enters the pad branch and applies fade_out. The paper's numbers depend on this behavior,
    so it must not be 'fixed' (use a separate pad_type if a corrected version is needed).
    """
    import torchaudio.transforms as T

    pp = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat_fade"))
    audio = torch.randn(1, 128000)
    out = pp(audio)
    expected = T.Fade(fade_in_len=0, fade_out_len=1000, fade_shape="linear")(audio)
    assert torch.equal(out, expected)
    assert not torch.equal(out, audio)


def test_repeat_fade_cuts_without_fade_when_longer():
    pp = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat_fade"))
    audio = torch.randn(1, 200000)
    assert torch.equal(pp(audio), audio[:, :128000])


def test_plain_repeat_stays_unfaded():
    """The existing 'repeat' behavior (no fade) must stay unchanged — protects other call sites."""
    pp = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat"))
    audio = torch.randn(1, 50_000)
    out = pp(audio)
    assert torch.equal(out, audio.repeat(1, 3)[:, :128000])
