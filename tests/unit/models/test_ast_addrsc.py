"""
ADD-RSC(AST 변형) P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_addrsc_goldens.py — 공식 deegy666/ADD-RSC models/{adapt_diff_denoise,ast,bias_denoise_loss}.py 를
timm 0.4.5 참조 venv 에서 실행 (난수 파라미터는 det_fill 로 결정화). 데이터 파이프라인은 patch-mix 골든이 이미 고정한다.
"""
import pathlib

import pytest
import torch
import torch.nn.functional as F

from rsc_toolkit.models.addrsc import DiffTransformerLayer, LabelSmoothingLoss, lambda_init_fn
from rsc_toolkit.models.ast_addrsc import ADDRSC


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "addrsc" / "model.pt"
CKPT = REPO / "pretrained_models/upstream/audioset_0.4593.pth"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="골든 없음")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="AudioSet ckpt 없음")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def test_denoise_layer_layout():
    layer = DiffTransformerLayer(256, 8, 6)
    assert abs(layer.attn.lambda_init - lambda_init_fn(6)) < 1e-12 and abs(lambda_init_fn(6) - 0.70081) < 1e-4
    assert sum(p.numel() for p in layer.parameters()) == 955204
    z, aux = layer(torch.randn(2, 1024, 256))
    assert z.shape == (2, 1024, 256) and aux.shape == (2, 4)
    # 보조 로짓 = 앞 두 프레임 평균의 Linear
    x1 = layer.aff(torch.randn(1, 1024, 256))
    assert torch.allclose(layer.classifier((x1[:, 0] + x1[:, 1]) / 2), layer.classifier(x1[:, :2].mean(1)))


def test_label_smoothing_matches_formula():
    crit = LabelSmoothingLoss(size=4, smoothing=0.2)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 3])
    true = torch.full((3, 4), 0.2 / 3).scatter_(1, y.unsqueeze(1), 0.8)
    ref = F.kl_div(torch.log_softmax(x, 1), true, reduction="sum") / 3
    assert torch.allclose(crit(x, y), ref, atol=1e-6)


@needs_golden
@needs_ckpt
def test_model_matches_reference():
    g = torch.load(GOLDEN, weights_only=False)
    m = ADDRSC(n_cls=4, pretrained_path=str(CKPT))
    assert m.grid == (64, 16) and list(m.v.pos_embed.shape) == g["pos_embed_shape"]
    assert {k: list(v.shape) for k, v in m.bias_denoise_encoder.state_dict().items()} == g["denoise_keys"]
    assert {k: list(v.shape) for k, v in m.v.state_dict().items()} == g["v_keys"]
    assert {k: list(v.shape) for k, v in m.classifier.state_dict().items()} == g["classifier_keys"]
    assert abs(m.bias_denoise_encoder.attn.lambda_init - g["lambda_init"]) < 1e-12
    # AudioSet 로드: pos_embed 만 모양 불일치(미로드), 나머지 v.*·mlp_head(527) 로드
    info = m.audioset_load_info  # 생성 시(527 헤드) 로드 결과 — 이후 mlp_head 는 n_cls 헤드로 대체됨
    assert info["mismatch"] == g["audioset_mismatch"] == ["module.v.pos_embed"]
    assert sorted(info["loaded"]) == g["audioset_loaded"]
    # 난수 파라미터 결정화 (골든과 동일)
    for p in m.bias_denoise_encoder.parameters():
        _det_fill(p)
    for p in m.classifier.parameters():
        _det_fill(p)
    _det_fill(m.v.pos_embed)
    m.eval()
    x = g["input"].squeeze(1)  # (2, 798, 128) = (time, freq)
    with torch.no_grad():
        images = m.to_model_input(x.transpose(-2, -1))  # 툴킷 fbank 배치 (B,128,798) 경로
        assert torch.allclose(images, g["images"].squeeze(1), atol=1e-6)  # torchvision Resize 와 동일
        output, aux = m.forward_all(images)
        denoise_images, _ = m.bias_denoise_encoder(images)
    assert torch.allclose(denoise_images, g["denoise_images"], rtol=1e-4, atol=1e-3), (denoise_images - g["denoise_images"]).abs().max()
    assert torch.allclose(aux, g["aux_logits"], rtol=1e-4, atol=1e-2), (aux - g["aux_logits"]).abs().max()
    assert torch.allclose(output, g["output"], rtol=1e-4, atol=1e-3), (output - g["output"]).abs().max()
    m.amp = False
    m.train()
    with torch.no_grad():
        out2, loss = m.training_forward(x.transpose(-2, -1), g["labels"], F.cross_entropy)
    assert torch.allclose(loss, g["loss"], rtol=1e-4), (loss, g["loss"])


@needs_ckpt
def test_training_forward_amp_and_grads():
    if not torch.cuda.is_available():
        pytest.skip("GPU 없음")
    m = ADDRSC(n_cls=4, pretrained_path=str(CKPT)).cuda().train()
    x = torch.randn(2, 128, 798, device="cuda")
    logits, loss = m.training_forward(x, torch.tensor([0, 3], device="cuda"), F.cross_entropy)
    assert logits.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert m.bias_denoise_encoder.aff.w1.grad is not None and m.classifier[1].weight.grad is not None
    assert m.mlp_head[1].weight.grad is None  # 원본처럼 forward 에서 미사용
