"""
Official-code parity golden tests for the 4 SFT backbones (ResNet18/34, EfficientNet-B0, CNN6).

Goldens were produced by running the official sgscl repository code in the reference environment (torch 1.13 + torchvision 0.14)
(scripts/reproduction/dump_sgscl_goldens.py). Measured: all state_dict keys have identical shapes,
forward outputs are bit-identical (max|diff| 0.0).

Original quirk: the 1-channel first conv of ResNet/EfficientNet is deleted from the ImageNet sd and
stays randomly initialized. For value comparison, both the dump and the test overwrite it with the same deterministic pattern
before comparing (does not affect verifying the structure or the loading of the remaining weights).
"""
import json
import pathlib

import pytest
import torch

from rsc_toolkit.processors.mel import MelBankAudioProcessor, generate_fbank


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "sgscl_sft"
PANN = any((REPO / c).exists() for c in (
    "pretrained_models/upstream/Cnn6_mAP=0.343.pth", "pretrained_models/Cnn6_mAP=0.343.pth"))

needs_golden = pytest.mark.skipif(
    not (GOLDEN / "state_dict_shapes.json").exists(),
    reason="golden missing — generate with scripts/reproduction/dump_sgscl_goldens.py",
)

CASES = [
    ("resnet18_sft", "resnet18", "conv1"),
    ("resnet34_sft", "resnet34", "conv1"),
    ("efficientnet_b0_sft", "efficientnet_b0", "stem"),
    ("cnn6_sft", "cnn6", None),
]


def _det_fill(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def _build(reg_name: str, det: str | None):
    from rsc_toolkit.models.factory import ModelFactory

    if reg_name == "cnn6_sft" and not PANN:
        pytest.skip("PANN CNN6 ckpt missing (pretrained_models/upstream/)")
    m = ModelFactory.get_model_class(reg_name)(n_cls=4, from_sl_official=True)
    if det == "conv1":
        _det_fill(m.backbone.conv1.weight)
    elif det == "stem":
        _det_fill(m.backbone.features[0][0].weight)
    m.eval()
    return m


@needs_golden
@pytest.mark.parametrize("reg_name,golden_name,det", CASES, ids=[c[0] for c in CASES])
def test_state_dict_matches_reference(reg_name, golden_name, det):
    m = _build(reg_name, det)
    golden = json.load(open(GOLDEN / "state_dict_shapes.json"))[golden_name]
    ours = {k[len("backbone."):]: list(v.shape) for k, v in m.state_dict().items() if k.startswith("backbone.")}
    extra_top = [k for k in m.state_dict() if not k.startswith(("backbone.", "classifier."))]

    assert not [k for k in golden if k not in ours], "keys present in the reference are missing"
    assert not {k: (golden[k], ours[k]) for k in golden if golden[k] != ours.get(k)}, "shape mismatch"
    assert not [k for k in ours if k not in golden], "backbone keys not present in the reference"
    assert not extra_top, f"keys outside backbone/classifier: {extra_top[:5]}"


@needs_golden
@pytest.mark.parametrize("reg_name,golden_name,det", CASES, ids=[c[0] for c in CASES])
def test_forward_features_match_reference(reg_name, golden_name, det):
    """
    Input in the toolkit layout (B, n_mels, frames) matches the reference output through the adapter.
    Measured (torch 1.13 reference vs 2.7): all 4 are bit-identical.
    """
    m = _build(reg_name, det)
    x_paper = torch.load(GOLDEN / "features_input.pt")          # (2, 1, 798, 128)
    x_toolkit = x_paper.squeeze(1).transpose(1, 2).contiguous()  # (2, 128, 798)
    ref = torch.load(GOLDEN / f"{golden_name}_features.pt")
    with torch.no_grad():
        got = m.forward_features(x_toolkit)
    assert torch.allclose(got, ref, atol=1e-5), (
        f"{reg_name}: max|diff|={float((got - ref).abs().max()):.3e}"
    )


def test_per_sample_fbank_normalization_matches_reference_formula():
    """
    Non-AST branch of sgscl generate_fbank: mean, std = fbank.mean(), fbank.std();
    (fbank - mean) / (std * 2). kaldi fbank itself is already pinned by the pipeline golden.
    """
    torch.manual_seed(0)
    audio = torch.randn(1, 32000)
    raw = generate_fbank(audio, sampling_rate=16000, n_mels=128, normalize=False)
    ours = generate_fbank(audio, sampling_rate=16000, n_mels=128, per_sample_stats=True)
    expected = (raw - raw.mean()) / (raw.std() * 2)
    assert torch.equal(ours, expected)


def test_per_sample_stats_changes_feature_cache_key():
    """Features with different normalization must not share a cache."""
    a = MelBankAudioProcessor(n_mels=128, sampling_rate=16000).get_processor_config()
    b = MelBankAudioProcessor(n_mels=128, sampling_rate=16000, per_sample_stats=True).get_processor_config()
    assert a != b and b["per_sample_stats"] is True
