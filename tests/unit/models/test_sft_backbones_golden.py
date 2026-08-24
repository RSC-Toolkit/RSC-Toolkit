"""
SFT 백본 4종(ResNet18/34, EfficientNet-B0, CNN6)의 P0 골든 테스트.

골든은 공식 sgscl 저장소 코드를 참조 환경(torch 1.13 + torchvision 0.14)에서 실행해
떴다 (scripts/reproduction/dump_sgscl_goldens.py). 실측: state_dict 전 키 동형,
forward 출력 비트 단위 동일 (max|diff| 0.0).

원본 quirk: ResNet/EfficientNet 의 1채널 첫 conv 는 ImageNet sd 에서 삭제돼
랜덤 초기화로 남는다. 값 비교를 위해 덤프/테스트 양쪽에서 같은 결정 패턴으로
덮은 뒤 대조한다 (구조·나머지 가중치 로딩의 동일성 검증엔 영향 없음).
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
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "sgscl_sft"
PANN = any((REPO / c).exists() for c in (
    "pretrained_models/upstream/Cnn6_mAP=0.343.pth", "pretrained_models/Cnn6_mAP=0.343.pth"))

needs_golden = pytest.mark.skipif(
    not (GOLDEN / "state_dict_shapes.json").exists(),
    reason="골든 없음 — scripts/reproduction/dump_sgscl_goldens.py 로 생성",
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
        pytest.skip("PANN CNN6 ckpt 없음 (pretrained_models/upstream/)")
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

    assert not [k for k in golden if k not in ours], "참조에 있는 키 누락"
    assert not {k: (golden[k], ours[k]) for k in golden if golden[k] != ours.get(k)}, "모양 불일치"
    assert not [k for k in ours if k not in golden], "참조에 없는 backbone 키"
    assert not extra_top, f"backbone/classifier 외 키: {extra_top[:5]}"


@needs_golden
@pytest.mark.parametrize("reg_name,golden_name,det", CASES, ids=[c[0] for c in CASES])
def test_forward_features_match_reference(reg_name, golden_name, det):
    """
    툴킷 레이아웃 (B, n_mels, frames) 입력이 어댑터를 거쳐 참조 출력과 일치하는지.
    실측(2026-08-24, torch 1.13 참조 vs 2.7): 4종 모두 비트 단위 동일.
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
    sgscl generate_fbank 의 비-AST 분기: mean, std = fbank.mean(), fbank.std();
    (fbank - mean) / (std * 2). kaldi fbank 자체는 파이프라인 골든으로 이미 고정됨.
    """
    torch.manual_seed(0)
    audio = torch.randn(1, 32000)
    raw = generate_fbank(audio, sampling_rate=16000, n_mels=128, normalize=False)
    ours = generate_fbank(audio, sampling_rate=16000, n_mels=128, per_sample_stats=True)
    expected = (raw - raw.mean()) / (raw.std() * 2)
    assert torch.equal(ours, expected)


def test_per_sample_stats_changes_feature_cache_key():
    """정규화 방식이 다른 피처가 같은 캐시를 쓰면 안 된다."""
    a = MelBankAudioProcessor(n_mels=128, sampling_rate=16000).get_processor_config()
    b = MelBankAudioProcessor(n_mels=128, sampling_rate=16000, per_sample_stats=True).get_processor_config()
    assert a != b and b["per_sample_stats"] is True
