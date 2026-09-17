"""
Official-code parity golden and configuration tests for Lung-SRAD (DASS-medium + SAR blur + Dual-Axis Patch-Mix CL).

Golden source: scripts/reproduction/dump_lungsrad_goldens.py — the official Lung-SRAD models_DASS/* (clone unmodified, CUDA selective_scan kernel) run
on GPU with the real ckpt (DASS_medium_v2.pth). Here the toolkit implementation runs on the pure torch fallback (CPU) or on GPU and is compared.
On CPU the random numbers the original draws from a CUDA generator (randint/randperm on device) cannot be reproduced, so torch.randint/torch.randperm are
pinned inside the tests only to return the lam/index/segments recorded in the golden (with a GPU, reproduced with the same call order as the original, without pinning).
"""
import os
import pathlib
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import torch.nn as nn

from rsc_toolkit.metrics.icbhi.binary_score import PaperBinaryScore
from rsc_toolkit.models.backbone.vmamba import (
    SS2D, VSSM, blur_separable, cuda_kernel_available, gaussian_kernel1d, selective_scan_torch,
)
from rsc_toolkit.models.backbone.vmamba.selective_scan import SelectiveScanOflex
from rsc_toolkit.models.dass_lungsrad import DASSLungSRAD


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "lungsrad" / "model.pt"
CKPT = REPO / "pretrained_models/upstream/dass/DASS_medium_v2.pth"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="DASS ckpt missing")
needs_kernel = pytest.mark.skipif(not (cuda_kernel_available() and torch.cuda.is_available()), reason="selective_scan CUDA kernel missing")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = False      # same exact fp32 path as the golden (same settings as the dump script)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# The CPU fallback of the 48M model grabs too many threads and pushes server load up to 300 — limit threads only while this file runs.
# **Do not set it at module top level**: pytest collects all modules first, so set_num_threads at that point would
# persist for the whole run, and torchaudio T.Resample gives bitwise-different values depending on the thread count
# (measured: the waveform bitwise comparison in tests/unit/datasets/test_sprsound_sclmeta.py fails only with 16 threads).


@pytest.fixture(autouse=True, scope="module")
def _limit_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("RSC_TEST_THREADS", "16")))
    yield
    torch.set_num_threads(prev)


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, weights_only=False)


def _build(method: str, **kw) -> DASSLungSRAD:
    torch.manual_seed(0)
    m = DASSLungSRAD(n_cls=4, method=method, pretrained_path=str(CKPT), **kw)
    _det_fill(m.v.classifier.head.weight)
    _det_fill(m.v.classifier.head.bias)
    return m.to(DEVICE).eval()


@pytest.fixture(scope="module")
def model_blur():
    return _build("patchmix_cl")


@contextmanager
def _fixed_rng(start=None, perm_first=None):
    """On CPU, pin randint/randperm(N, device) of the original (CUDA generator) to the golden values. randperm(B) (CPU, no device argument) is left as is."""
    if DEVICE.type == "cuda":
        yield
        return
    orig_randint, orig_randperm = torch.randint, torch.randperm

    def randint(*a, **k):
        return torch.tensor([int(start)]) if start is not None else orig_randint(*a, **k)

    def randperm(n, *a, **k):
        if perm_first is not None and "device" in k:
            rest = [i for i in range(n) if i not in set(perm_first.tolist())]
            return torch.tensor(perm_first.tolist() + rest)
        return orig_randperm(n, *a, **k)

    torch.randint, torch.randperm = randint, randperm
    try:
        yield
    finally:
        torch.randint, torch.randperm = orig_randint, orig_randperm


# ------------------------------------------------------------------------------------------ configuration
def test_gaussian_kernel_values():
    g = gaussian_kernel1d(5, 3.0)
    assert torch.allclose(g, torch.tensor([0.1782, 0.2105, 0.2225, 0.2105, 0.1782]), atol=5e-5) and abs(g.sum() - 1) < 1e-6
    x = torch.randn(2, 3, 8, 8)
    y = blur_separable(x, 5, 3.0)
    assert y.shape == x.shape and torch.allclose(y.mean(), x.mean(), atol=0.2)


def test_ss2d_layout_and_blur_hook():
    op = SS2D(d_model=96, d_state=1, ssm_ratio=1.0, d_conv=3, conv_bias=False, forward_type="v05_noz", channel_first=True,
              blur_enable=True, blur_kernel_size=5, blur_sigma=3.0, ssm_backend="torch")
    assert op.disable_z and op.in_proj.weight.shape == (96, 96) and op.A_logs.shape == (384, 1) and op.x_proj_weight.shape == (4, 8, 96)
    x = torch.randn(1, 96, 8, 16)
    op.eval()
    with torch.no_grad():
        y_blur = op(x)
        op.blur_enable = False
        y_noblur = op(x)
    assert y_blur.shape == x.shape and y_noblur.shape == x.shape
    assert not torch.allclose(y_blur, y_noblur)  # the blur hook actually changes the output


def test_vssm_medium_structure_and_blur_blocks():
    m = DASSLungSRAD(n_cls=4, method="ce_blur", audioset_pretrain=False, ssm_backend="torch")
    on = [(i, j) for i, layer in enumerate(m.v.layers) for j, blk in enumerate(layer.blocks) if blk.op.blur_enable]
    assert on == [(2, 2), (2, 3)]
    assert [len(l.blocks) for l in m.v.layers] == [2, 2, 20, 2] and m.v.dims == [96, 192, 384, 768]
    assert sum(p.numel() for p in m.v.parameters()) == 48246052  # golden n_params (including the 4-class head; 48.246M)
    assert isinstance(m.projector, nn.Identity)
    ft = DASSLungSRAD(n_cls=4, method="ce", audioset_pretrain=False, ssm_backend="torch")
    assert not any(blk.op.blur_enable for l in ft.v.layers for blk in l.blocks)
    assert m.v.patch_embed[0].weight.shape == (48, 1, 3, 3)


@needs_ckpt
def test_ckpt_keys_match():
    m = DASSLungSRAD(n_cls=4, method="ce", pretrained_path=str(CKPT), ssm_backend="torch")
    assert m.dass_load_info["missing"] == ["classifier.head.bias", "classifier.head.weight"] and m.dass_load_info["unexpected"] == []
    assert m.dass_load_info["loaded"] == 490


# ------------------------------------------------------------------------------------------ golden comparison
@needs_golden
@needs_ckpt
def test_resize_and_patch_embed_match(golden, model_blur):
    g = golden
    raw = g["input_raw"].transpose(1, 2).to(DEVICE)          # (2,128,798) toolkit layout
    inp = model_blur.to_model_input(raw)
    assert inp.shape == (2, 1, 128, 1024)
    assert torch.allclose(inp.cpu(), g["model_input_bchw"][:2], atol=1e-5)
    x4 = g["model_input_bchw"].to(DEVICE)
    pe = model_blur.v.patch_embed(x4)
    assert pe.shape == (4, 96, 32, 256)
    assert torch.allclose(pe.cpu(), g["patch_embed"], atol=1e-4)
    assert torch.allclose(golden["blur_kernel"], gaussian_kernel1d(5, 3.0), atol=1e-6)


@needs_golden
@needs_ckpt
def test_forward_matches_reference(golden, model_blur):
    g = golden
    x4 = g["model_input_bchw"].to(DEVICE)
    with torch.no_grad():
        h = model_blur.v.patch_embed(x4)
        for i, layer in enumerate(model_blur.v.layers):
            h = layer(h)
            if i in g["stage_outputs_blur"]:
                ref = g["stage_outputs_blur"][i]
                hc = h.cpu()[:, :, :, :ref.shape[-1]]
                assert torch.allclose(hc, ref, atol=1e-3, rtol=1e-3), f"stage {i} max diff {(hc-ref).abs().max()}"
        feats, logits = model_blur.forward_features(x4)
    assert torch.allclose(feats.cpu(), g["features_blur"], atol=1e-3), (feats.cpu() - g["features_blur"]).abs().max()
    assert torch.allclose(logits.cpu(), g["logits_blur"], atol=1e-3), (logits.cpu() - g["logits_blur"]).abs().max()
    # blur off (FT structure) — same weights, blur blocks ∅
    ft = _build("ce")
    with torch.no_grad():
        feats_off, logits_off = ft.forward_features(x4)
    assert torch.allclose(feats_off.cpu(), g["features_noblur"], atol=1e-3)
    assert torch.allclose(logits_off.cpu(), g["logits_noblur"], atol=1e-3)
    assert not torch.allclose(g["logits_noblur"], g["logits_blur"], atol=1e-3)  # blur actually changes the output
    # logits of training_forward (eval mode) = forward_features logits, loss = CE
    with torch.no_grad():
        images = g["images"].transpose(1, 2).to(DEVICE)  # (4,128,1024)
        out, loss = model_blur.training_forward(images, g["labels"].to(DEVICE), nn.CrossEntropyLoss())
    if DEVICE.type == "cpu":
        assert torch.allclose(out.cpu(), g["logits_blur"], atol=1e-3) and abs(float(loss) - g["loss"]["ce"]) < 1e-3
    else:  # GPU uses the autocast fp16 path -> compared with the original fp16 golden (loose)
        assert torch.allclose(out.cpu(), g["logits_blur_fp16"], atol=5e-2)


@needs_golden
@needs_ckpt
@pytest.mark.parametrize("mix_type", ["time", "freq", "2d"])
def test_patchmix_as_code_matches_reference(golden, model_blur, mix_type):
    g = golden["patchmix"][mix_type]
    x4 = golden["model_input_bchw"].to(DEVICE)
    pe_ref = golden["patch_embed"]
    dims = {d: g["changed_dims"][f"dim{d}"] for d in (1, 2, 3)}
    C, Fq, T = pe_ref.shape[1:]
    # original behavior (as_code): time -> contiguous segment on dim2 (frequency 32), all channels and all time; freq -> contiguous segment on dim1 (channels 96); 2d -> full time range, (channel×frequency) tokens
    if mix_type == "time":
        seg = dims[2]
        assert len(dims[1]) == C and len(dims[3]) == T and 1 <= len(seg) < Fq and (seg.max() - seg.min() + 1) == len(seg)
        assert abs(g["lam"] - (1 - len(seg) / Fq)) < 1e-9
        fixed = dict(start=int(seg.min()))
    elif mix_type == "freq":
        seg = dims[1]
        assert len(dims[2]) == Fq and len(dims[3]) == T and 1 <= len(seg) < C and (seg.max() - seg.min() + 1) == len(seg)
        assert abs(g["lam"] - (1 - len(seg) / C)) < 1e-9
        fixed = dict(start=int(seg.min()))
    else:
        assert len(dims[3]) == T
        tok = g["changed_tokens_cf"]   # (C·F) token index
        assert len(tok) > 0 and len(tok) < C * Fq
        fixed = dict(perm_first=tok)
    # reproduction: same seed and call order (GPU) / pinned to golden segments (CPU)
    with torch.no_grad(), _fixed_rng(**fixed):
        np.random.seed(g["seed"])
        torch.manual_seed(g["seed"])
        pe = model_blur.v.patch_embed(x4)
        pe0 = pe.detach().clone()
        xm, y_a, y_b, lam, index = model_blur.v.apply_patch_mix(pe, golden["labels"].to(DEVICE), mix_type)
    assert torch.equal(index.cpu(), g["index"]) and torch.equal(y_b.cpu(), g["y_b"])
    assert abs(lam - g["lam"]) < 1e-9
    assert abs(float(xm.float().sum()) - g["mixed_patch_embed_sum"]) < 1e-2 * max(1.0, abs(g["mixed_patch_embed_sum"]))
    assert torch.allclose(xm.float().cpu().flatten()[::997], g["mixed_patch_embed_checksum"], atol=1e-4)
    ch = (xm != pe0).cpu()   # changed positions are compared with the original computed on the same device (the golden pe differs at ~1e-5, so everything would look 'changed')
    for d in (1, 2, 3):
        assert torch.equal(ch.any(dim=tuple(k for k in range(4) if k != d)).nonzero().flatten(), dims[d])
    with torch.no_grad(), _fixed_rng(**fixed):
        np.random.seed(g["seed"])
        torch.manual_seed(g["seed"])
        mf, _, _, lam2, index2 = model_blur.v.forward_features(x4, y=golden["labels"].to(DEVICE), patch_mix=True, mix_type=mix_type)
    assert abs(lam2 - lam) < 1e-12 and torch.equal(index2.cpu(), index.cpu())
    assert torch.allclose(mf.cpu(), g["mixed_features"], atol=2e-3), (mf.cpu() - g["mixed_features"]).abs().max()


def test_patchmix_as_paper_axes():
    torch.manual_seed(0)
    np.random.seed(0)
    v = VSSM(depths=[1, 1, 1, 1], dims=[8, 16, 32, 64], ssm_d_state=1, ssm_ratio=1.0, forward_type="v05_noz", norm_layer="LN2D",
             downsample_version="v3", patchembed_version="v2", ssm_backend="torch", mix_layout="as_paper", mix_beta=1.0)
    B, C, Fq, T = 3, 8, 6, 20
    x = torch.randn(B, C, Fq, T)
    y = torch.tensor([0, 1, 2])
    for mix_type, axis, size in (("time", 3, T), ("freq", 2, Fq)):
        for _ in range(5):
            xm, y_a, y_b, lam, index = v.apply_patch_mix(x.clone(), y, mix_type)
            changed = xm != x
            seg = changed.any(dim=tuple(k for k in range(4) if k != axis)).nonzero().flatten()
            others = [d for d in (1, 2, 3) if d != axis]
            if len(seg) == 0:  # identity index means no change
                assert torch.equal(index, torch.arange(B))
                continue
            assert (seg.max() - seg.min() + 1) == len(seg) and len(seg) < size
            for d in others:
                assert changed.any(dim=tuple(k for k in range(4) if k != d)).sum() == x.shape[d]
            assert abs(lam - (1 - len(seg) / size)) < 1e-9
            assert torch.equal(y_b, y[index])
    # 2d(as_paper): (F×T) patches, all channels
    xm, *_ = v.apply_patch_mix(x.clone(), y, "2d")
    changed = xm != x
    assert changed.any(dim=(0, 2, 3)).sum() in (0, C)
    # axes differ from as_code
    v.mix_layout = "as_code"
    xm2, *_ = v.apply_patch_mix(x.clone(), y, "time")
    ch2 = xm2 != x
    assert ch2.any(dim=(0, 1, 3)).sum() < Fq or ch2.sum() == 0   # as_code time = partial segment on dim2


@needs_golden
@needs_ckpt
def test_dual_loss_matches_reference(golden, model_blur):
    g = golden
    model_blur.projector.load_state_dict({k: v.to(DEVICE) for k, v in g["projector_state"].items()})
    x4 = g["model_input_bchw"].to(DEVICE)
    labels = g["labels"].to(DEVICE)
    L = g["loss"]
    seg_t = g["patchmix"]["time"]["changed_dims"]["dim2"]
    seg_f = g["patchmix"]["freq"]["changed_dims"]["dim1"]
    with torch.no_grad():
        feats, logits = model_blur.forward_features(x4)
        ce = nn.CrossEntropyLoss()(logits, labels)
        proj1 = model_blur.projector(feats)
        assert torch.allclose(proj1.cpu(), L["proj1"], atol=2e-3), (proj1.cpu() - L["proj1"]).abs().max()
        with _fixed_rng(start=int(seg_t.min())):
            np.random.seed(11); torch.manual_seed(11)
            lt = model_blur._axis_cl(x4, labels, proj1, "time")
        with _fixed_rng(start=int(seg_f.min())):
            np.random.seed(22); torch.manual_seed(22)
            lf = model_blur._axis_cl(x4, labels, proj1, "freq")
    assert abs(float(ce) - L["ce"]) < 1e-3 and abs(float(lt) - L["time"]) < 1e-3 and abs(float(lf) - L["freq"]) < 1e-3
    assert abs(float(ce + lt + lf) - L["total"]) < 2e-3
    # PatchMixConLoss alone
    u = g["conloss_unit"]
    val = model_blur.patchmix_con_loss(u["p1"].to(DEVICE), u["p2"].to(DEVICE), torch.tensor([0, 1, 2, 3]).to(DEVICE), None, u["lam"], u["index"].to(DEVICE))
    assert abs(float(val) - u["value"]) < 1e-5


def test_training_forward_paths_small():
    """Without pretraining, training_forward of ce / ce_blur / patchmix_cl(dual, time, freq, 2d) runs on small inputs and gradients flow."""
    torch.manual_seed(0)
    x = torch.randn(4, 128, 100)
    y = torch.tensor([0, 1, 2, 3])
    combos = (("ce", "dual", "as_code"), ("ce_blur", "dual", "as_code"), ("patchmix_cl", "dual", "as_code"), ("patchmix_cl", "2d", "as_code"),
              ("patchmix_cl", "time", "as_paper"), ("patchmix_cl", "freq", "as_paper"))
    for method, cl_mode, layout in combos:
        if True:
            m = DASSLungSRAD(n_cls=4, method=method, cl_mode=cl_mode, mix_layout=layout, audioset_pretrain=False, resize_to=(128, 128),
                             ssm_backend="torch", amp=False).train()
            logits, loss = m.training_forward(x, y, nn.CrossEntropyLoss())
            assert logits.shape == (4, 4) and torch.isfinite(loss)
            loss.backward()
            assert m.v.patch_embed[0].weight.grad is not None
            if method == "patchmix_cl":
                assert m.projector.linear1.weight.grad is not None
    m.eval()
    with torch.no_grad():
        logits, loss = m.training_forward(x, y, nn.CrossEntropyLoss())
    assert torch.allclose(loss, nn.CrossEntropyLoss()(logits, y))  # eval mode uses CE only


# ------------------------------------------------------------------------------------------ kernel vs fallback
@needs_kernel
def test_selective_scan_fallback_matches_kernel():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    for dt in (torch.float32, torch.float16):
        B, K, d, N, L = 2, 4, 96, 1, 32 * 256
        u = torch.randn(B, K * d, L, device=dev, dtype=dt, requires_grad=True)
        delta = (0.1 * torch.randn(B, K * d, L, device=dev, dtype=dt)).requires_grad_(True)
        A = (-torch.rand(K * d, N, device=dev) * 2 - 0.1).requires_grad_(True)
        Bs = torch.randn(B, K, N, L, device=dev, dtype=dt, requires_grad=True)
        Cs = torch.randn(B, K, N, L, device=dev, dtype=dt, requires_grad=True)
        Ds = torch.randn(K * d, device=dev, requires_grad=True)
        db = torch.randn(K * d, device=dev, requires_grad=True)
        y1 = SelectiveScanOflex.apply(u, delta, A, Bs, Cs, Ds, db, True, -1, -1, True)
        y2 = selective_scan_torch(u, delta, A, Bs, Cs, Ds, db, True, True)
        assert y1.dtype == y2.dtype == torch.float32
        assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4), (y1 - y2).abs().max()
        gr = torch.randn_like(y1)
        g1 = torch.autograd.grad(y1, (u, delta, A, Bs, Cs, Ds, db), gr)
        g2 = torch.autograd.grad(y2, (u, delta, A, Bs, Cs, Ds, db), gr)
        for a, b in zip(g1, g2):
            assert ((a - b).abs().max() / (a.abs().max() + 1e-12)) < (1e-4 if dt == torch.float32 else 1e-2)


@needs_kernel
@needs_ckpt
def test_model_kernel_vs_fallback_forward():
    torch.manual_seed(0)
    x = torch.randn(2, 128, 798, device="cuda")
    mk = DASSLungSRAD(n_cls=4, method="ce_blur", pretrained_path=str(CKPT), ssm_backend="cuda_oflex", amp=False).cuda().eval()
    mt = DASSLungSRAD(n_cls=4, method="ce_blur", pretrained_path=str(CKPT), ssm_backend="torch", amp=False).cuda().eval()
    mt.load_state_dict(mk.state_dict())
    with torch.no_grad():
        a, b = mk(x), mt(x)
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


# ------------------------------------------------------------------------------------------ 2-class derived metric
def test_two_class_derived_metric_matches_main_py():
    """main.py:420-427 --two_cls_eval: normal is correct if pred==0, abnormal if pred>0 -> Sp/Se/Score."""
    logits = torch.tensor([[2., 0, 0, 0], [0, 2., 0, 0], [0, 0, 2., 0], [0, 0, 0, 2.], [0, 2., 0, 0], [2., 0, 0, 0], [0, 0, 0, 2.], [0, 0, 2., 0]])
    labels = torch.tensor([0, 1, 2, 3, 3, 1, 0, 1])
    hits, counts = [0.0] * 4, [0.0] * 4
    preds = logits.argmax(1)
    for i in range(len(labels)):
        counts[labels[i].item()] += 1
        if labels[i] == 0 and preds[i] == 0:
            hits[0] += 1
        elif labels[i] != 0 and preds[i] > 0:
            hits[labels[i].item()] += 1
    sp = hits[0] / counts[0] * 100
    se = sum(hits[1:]) / sum(counts[1:]) * 100
    m = PaperBinaryScore(normal_index=0)
    m.update(logits, labels)
    out = m.compute()
    assert abs(float(out["paper_S_p"]) - sp) < 1e-4 and abs(float(out["paper_S_e"]) - se) < 1e-4
    assert abs(float(out["paper_score"]) - (sp + se) / 2) < 1e-4
