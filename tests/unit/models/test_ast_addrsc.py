"""
Official-code parity golden and configuration tests for ADD-RSC (AST variant).

Golden source: scripts/reproduction/dump_addrsc_goldens.py — the official deegy666/ADD-RSC models/{adapt_diff_denoise,ast,bias_denoise_loss}.py
run in the timm 0.4.5 reference venv (random parameters made deterministic with det_fill). The data pipeline is already pinned by the patch-mix goldens.
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
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "addrsc" / "model.pt"
CKPT = REPO / "pretrained_models/upstream/audioset_0.4593.pth"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="AudioSet ckpt missing")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def test_denoise_layer_layout():
    layer = DiffTransformerLayer(256, 8, 6)
    assert abs(layer.attn.lambda_init - lambda_init_fn(6)) < 1e-12 and abs(lambda_init_fn(6) - 0.70081) < 1e-4
    assert sum(p.numel() for p in layer.parameters()) == 955204
    z, aux = layer(torch.randn(2, 1024, 256))
    assert z.shape == (2, 1024, 256) and aux.shape == (2, 4)
    # auxiliary logits = Linear over the mean of the first two frames
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
    # AudioSet load: only pos_embed has a shape mismatch (not loaded); the remaining v.* and mlp_head (527) are loaded
    info = m.audioset_load_info  # load result at construction (527 head) — mlp_head is later replaced by the n_cls head
    assert info["mismatch"] == g["audioset_mismatch"] == ["module.v.pos_embed"]
    assert sorted(info["loaded"]) == g["audioset_loaded"]
    # make random parameters deterministic (same as the golden)
    for p in m.bias_denoise_encoder.parameters():
        _det_fill(p)
    for p in m.classifier.parameters():
        _det_fill(p)
    _det_fill(m.v.pos_embed)
    m.eval()
    x = g["input"].squeeze(1)  # (2, 798, 128) = (time, freq)
    with torch.no_grad():
        images = m.to_model_input(x.transpose(-2, -1))  # toolkit fbank batch (B,128,798) path
        assert torch.allclose(images, g["images"].squeeze(1), atol=1e-6)  # same as torchvision Resize
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
        pytest.skip("no GPU")
    m = ADDRSC(n_cls=4, pretrained_path=str(CKPT)).cuda().train()
    x = torch.randn(2, 128, 798, device="cuda")
    logits, loss = m.training_forward(x, torch.tensor([0, 3], device="cuda"), F.cross_entropy)
    assert logits.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert m.bias_denoise_encoder.aff.w1.grad is not None and m.classifier[1].weight.grad is not None
    assert m.mlp_head[1].weight.grad is None  # unused in forward, as in the original
