"""
Official-code parity golden and configuration tests for LoRA-RSC (ICASSP 2025).

Golden source: scripts/reproduction/dump_lorarsc_goldens.py — the official deegy666/LoRA-RSC src/AST_LoRA_ICBHI.py, src/AST.py, dataset/icbhi.py chain
run in a reference venv with the README-pinned environment (transformers 4.28.1) (LoRA and head made deterministic with det_fill).
"""
import pathlib

import pytest
import torch
import torch.nn.functional as F

from rsc_toolkit.models.ast_lora import ASTLoRARSC, LoRALinear
from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor
from rsc_toolkit.processors.mel import MelBankAudioProcessor


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "lorarsc"
HF = REPO / "pretrained_models/upstream/hf/ast-finetuned-audioset-10-10-0.4593"
DATA = REPO / "sample_data/toy_dataset/files"
needs_golden = pytest.mark.skipif(not (GOLDEN / "model.pt").exists(), reason="golden missing")
needs_hf = pytest.mark.skipif(not (HF / "config.json").exists(), reason="HF AST snapshot missing")


def _det_fill(t, scale=1.0):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape) * scale)


def _inject_golden(m: ASTLoRARSC, g: dict):
    for i, layer in enumerate(m.ast.layers):
        for t in ("k", "v"):
            mod = getattr(layer.attention, f"{t}_proj")
            with torch.no_grad():
                mod.lora_down.weight.copy_(g["lora_state"][f"encoder.layer.{i}.attention.attention.lora_down_{t}.weight"])
                mod.lora_up.weight.copy_(g["lora_state"][f"encoder.layer.{i}.attention.attention.lora_up_{t}.weight"])
    m.classification_head.load_state_dict(g["head_state"])


@needs_golden
def test_features_match_hf_extractor_and_sclmeta_chain():
    """Toolkit sclmeta padding + fbank(128, target_frames 500) == original waveform chain + 4.28.1 ASTFeatureExtractor(max_length 500)."""
    g = torch.load(GOLDEN / "features.pt", weights_only=False)
    import torchaudio
    import torchaudio.transforms as T

    pre = AudioPreprocessor(16000, AudioPreprocessConfig(8.0, "sclmeta"))
    fp = MelBankAudioProcessor(n_mels=128, sampling_rate=16000, target_frames=500)
    for i, meta in enumerate(g["meta"]):
        wav = DATA / f"{meta['file']}.wav"
        sr = torchaudio.info(str(wav)).sample_rate
        # same procedure as IndexDataset(resample_after_slice=True)
        audio, _ = torchaudio.load(str(wav), frame_offset=int(meta["onset"] * sr), num_frames=int(meta["offset"] * sr) - int(meta["onset"] * sr))
        audio = T.Resample(sr, 16000)(audio)
        audio = pre(audio)
        assert torch.allclose(audio, g["waveforms"][i : i + 1], atol=1e-6)
        feat = fp.make_features(audio.squeeze(0))  # (128, 500)
        ref = g["features_noise"][i].transpose(0, 1)
        assert feat.shape == (128, 500)
        assert torch.allclose(feat, ref, atol=1e-4), float((feat - ref).abs().max())
    # the original noise (scalar DC offset per clip) is cancelled by kaldi remove_dc_offset — feature difference is at float precision
    assert float((g["features_noise"] - g["features_clean"]).abs().max()) < 1e-4
    assert all(m["noise_shape"] == [1] for m in g["meta"])


@needs_hf
def test_layout_and_param_counts():
    m = ASTLoRARSC()
    assert m.lora_rank == 8 and abs(m.lora_scaling - 8 / 96) < 1e-12
    assert m.num_trainable() == 299_524                       # README 0.299M (truncated)
    assert sum(p.numel() for p in m.parameters()) == 85_711_108 + 12 * 2 * 2 * 768 * 8
    assert float(m.ast.embeddings.position_embeddings.abs().max()) == 0.0 and not m.ast.embeddings.position_embeddings.requires_grad
    assert tuple(m.ast.embeddings.position_embeddings.shape) == (1, 590, 768)
    assert isinstance(m.ast.layers[0].attention.k_proj, LoRALinear) and not isinstance(m.ast.layers[0].attention.q_proj, LoRALinear)
    # 4.28.1 fast-init quirk: initial LoRA delta != 0 (down is not 0)
    assert float(m.ast.layers[0].attention.k_proj.lora_down.weight.abs().max()) > 0
    # original train() override: backbone stays in eval
    m.train()
    assert not m.ast.layers[0].training and m.ast.layernorm.training and m.classification_head.training
    # a 798-frame input uses only the first 500
    x = torch.randn(2, 128, 798)
    assert torch.equal(m.to_model_input(x), x.transpose(1, 2)[:, :500])
    full = ASTLoRARSC(method="full_ft")
    assert full.num_trainable() == 85_711_108                # disagrees with README 85.74M (the code configuration gives 85.71M)
    paper = ASTLoRARSC(lora_init="paper")
    assert float(paper.ast.layers[0].attention.k_proj.lora_down.weight.abs().max()) == 0.0


@needs_golden
@needs_hf
def test_model_matches_reference():
    g = torch.load(GOLDEN / "model.pt", weights_only=False)
    assert g["pos_embed_absmax"] == 0.0 and not g["pos_embed_requires_grad"]
    assert g["n_train"] == 299_524 and g["rank_dim"] == 8 and abs(g["scaling"] - 8 / 96) < 1e-12
    # in the reference code LoRA down is not 0 right after from_pretrained (trunc_normal 0.02 re-init) — confirms the quirk
    s = g["lora_init_stats_layer0"]
    assert s["lora_down_k"]["absmax"] > 0 and 0.01 < s["lora_down_k"]["std"] < 0.03
    m = ASTLoRARSC()
    assert m.num_trainable() == g["n_train"] and sum(p.numel() for p in m.parameters()) == g["n_total"]
    _inject_golden(m, g)
    m.eval()
    x = g["input"].squeeze(1).transpose(1, 2)  # (N,500,128) -> toolkit (N,128,500)
    with torch.no_grad():
        logits = m(x)
        _, loss = m.training_forward(x, g["labels"], F.cross_entropy)
    assert torch.allclose(logits, g["logits"], rtol=1e-4, atol=1e-4), float((logits - g["logits"]).abs().max())
    assert torch.allclose(loss, g["loss"], rtol=1e-4, atol=1e-4), (float(loss), float(g["loss"]))
    m.train()
    with torch.no_grad():
        logits_tr = m(x)
    assert torch.allclose(logits_tr, g["logits_train_mode"], rtol=1e-4, atol=1e-4)
    # full-FT
    f = ASTLoRARSC(method="full_ft")
    f.classification_head.load_state_dict(g["full_ft"]["head_state"])
    assert f.num_trainable() == g["full_ft"]["n_train"]
    f.eval()
    with torch.no_grad():
        lf = f(x)
    assert torch.allclose(lf, g["full_ft"]["logits"], rtol=1e-4, atol=1e-4), float((lf - g["full_ft"]["logits"]).abs().max())


@needs_hf
def test_grads_only_on_trainable():
    m = ASTLoRARSC().train()
    x = torch.randn(2, 128, 500)
    logits, loss = m.training_forward(x, torch.tensor([0, 3]), F.cross_entropy)
    loss.backward()
    att = m.ast.layers[0].attention
    assert att.k_proj.lora_down.weight.grad is not None and att.v_proj.lora_up.weight.grad is not None
    assert att.k_proj.base.weight.grad is None and att.q_proj.weight.grad is None
    assert m.ast.layernorm.weight.grad is not None and m.classification_head.weight.grad is not None


def test_lorarsc_noise_dataset_is_deterministic_per_sample():
    """1D noise interpretation (rsc_toolkit/datasets/lorarsc_noise.py): per-sample N(0,0.3²), deterministic by sample_id (cached once, shared by train/test), original float64->float32 order."""
    from rsc_toolkit.datasets.lorarsc_noise import LoRARSCNoisyIndexDataset as D

    torch.manual_seed(0)
    audio = torch.randn(1, 128000) * 0.05
    a1 = D.add_noise(audio, "rec_x_cycle_1")
    a2 = D.add_noise(audio, "rec_x_cycle_1")
    a3 = D.add_noise(audio, "rec_x_cycle_2")
    assert a1.dtype == torch.float32 and a1.shape == audio.shape
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, a3)
    delta = (a1 - audio).double()
    assert abs(delta.std().item() - 0.3) < 0.01 and abs(delta.mean().item()) < 0.01
    # unlike the scalar DC offset (public-code interpretation), this noise is not removed per frame
    assert (delta - delta.mean()).abs().max() > 0.5
