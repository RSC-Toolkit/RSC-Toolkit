"""
Patch-Mix 데이터 파이프라인의 P0 골든 테스트.

골든은 저자 코드(get_individual_cycles_torchaudio → cut_pad_sample_torchaudio →
generate_fbank, SpecAugment)를 같은 torchaudio 위에서 실행해 떴다
(scripts/reproduction/dump_patchmix_pipeline_goldens.py — 원 저장소는 torchaudio
버전을 고정하지 않으므로 기준은 코드 자체다).

여기서 고정하는 우리 체인:
  load_audio(fade=True)  ==  전체 녹음 fade(sr/16, linear)
  slice_audio            ==  _slice_data_torchaudio
  AudioPreprocessor(pad_type="repeat_fade")  ==  cut_pad_sample_torchaudio
  processors.mel.generate_fbank              ==  저자 generate_fbank (방향만 다름)
  augmentations.SpecAugment (freq-first)     ==  저자 SpecAugment (time-first)

SpecAugment 골든은 결함 #9 의 판정 근거이기도 하다: p=1.0 에서도
`p >= torch.randn(1)` 게이트로 약 16% 스킵되는 동작은 **원본에 있는 것**이며,
우리 이식이 같은 RNG 소비 순서로 같은 출력을 냄을 보인다.
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
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "patchmix_pipeline"
DATA = REPO / "sample_data" / "toy_dataset" / "files"

needs_golden = pytest.mark.skipif(
    not (GOLDEN / "fbank_golden.pt").exists(),
    reason="골든 없음 — scripts/reproduction/dump_patchmix_pipeline_goldens.py 로 생성",
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
            # fade_out 이 실제로 걸렸는지 (끝 1000 샘플 평균 절대값까지 일치)
            assert float(audio[..., -1000:].abs().mean()) == pytest.approx(fp["tail_mean_abs"], abs=1e-6)

            ours = generate_fbank(audio, sampling_rate=16000, n_mels=128)  # (798, 128)
            ref = golden[i]
            diff = float((ours - ref).abs().max())
            assert torch.allclose(ours, ref, atol=1e-5), (
                f"{entry['filename']} cycle{i}: fbank 불일치 max|diff|={diff:.3e}"
            )
            i += 1
    assert i == golden.shape[0]


@needs_golden
@pytest.mark.exact_golden  # 비트 단위 비교 — 개발 머신 전용 (CPU/BLAS/스레드 수가 다르면 마지막 자리에서 어긋난다)
def test_specaugment_matches_reference_with_same_seeds():
    """
    같은 seed 에서 저자 SpecAugment([c, time, freq] 입력, 내부 transpose)와
    우리 SpecAugment(freq-first 입력)가 동일한 결과를 낸다.
    RNG 소비 순서(torch.randn 게이트 → np.uniform/randint 마스크)가 같아야 성립한다.
    """
    img = torch.load(GOLDEN / "specaug_input.pt")  # [1, 798, 128] (time-first)
    ref_out = torch.load(GOLDEN / "specaug_golden.pt")  # [1, 798, 128]

    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)

    ours_in = img.squeeze(0).transpose(0, 1).clone()  # (128, 798) freq-first
    ours_out = SpecAugment(policy="icbhi_ast_sup", mask="mean")(ours_in)

    assert torch.equal(ours_out.transpose(0, 1), ref_out.squeeze(0))
    # 실제로 마스킹이 일어났는지 (gate 통과 seed 로 덤프됨)
    assert not torch.equal(ours_out, ours_in)


def test_repeat_fade_applies_fade_even_when_length_matches():
    """
    원본 cut_pad_sample_torchaudio 의 quirk: 길이가 정확히 목표와 같아도
    pad 분기로 들어가 fade_out 이 적용된다. 이 동작은 논문 수치의 전제이므로
    '고치면' 안 된다 (수정판이 필요하면 별도 pad_type 으로).
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
    """기존 'repeat' 의 동작(fade 없음)은 그대로여야 한다 — 다른 사용처 보호."""
    pp = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "repeat"))
    audio = torch.randn(1, 50_000)
    out = pp(audio)
    assert torch.equal(out, audio.repeat(1, 3)[:, :128000])
