"""
Lung-SRAD (RSC-Toolkit/Lung-SRAD d28d783) official-code parity goldens — runs the official models_DASS/{ast_models,vmamba_patchmix,csms6s,csm_triton}.py +
method/patchmix_cl.py + models/projector.py with the **real ckpt (DASS_medium_v2.pth)** on GPU and saves the results.

  # kernel (optional): build outputs/reference/dass_upstream/kernels/selective_scan into scratch (README procedure; not installed into .venv)
  #   cp -r kernels/selective_scan $S/ss_src && (remove the wheel import from setup.py) && MAX_JOBS=6 .venv/bin/python setup.py build_ext --build-lib $S/pylib
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$S/pylib:$S/shims .venv/bin/python scripts/reproduction/dump_lungsrad_goldens.py \
      --repo outputs/reference/lungsrad --ckpt pretrained_models/upstream/dass/DASS_medium_v2.pth --out tests/golden/lungsrad [--no-kernel]

The public code does not run as-is. This script does not import main.py and uses only the model and loss modules, so the clone is **not modified**.
Minimal patches needed to also train with main.py (documented; not needed for these goldens):
  (P1) main.py:27   `from method import PatchMixLoss, PatchMixConLoss` → `from method.patchmix_cl import PatchMixConLoss` (PatchMixLoss does not exist in the repository)
  (P2) util/icbhi_util.py:168  `args.meta_mode` → `getattr(args, "meta_mode", None)` (not defined in argparse)
  (P3) FT baseline (no --gaussian_blur): the plain vmamba.VSSM selected by ast_models.py:278-283 accepts only forward(x), so the kwargs from DASS.forward raise TypeError
       → route through vmamba_patchmix.VSSM with an empty blur block set via DASS(gaussian_blur=True, blur_blocks=set()) (same structure and weights). This script does the same.
  (P4) the ckpt path './pretrained_models/DASS_medium_v2.pth' is hard-coded as a relative path → symlink in cwd (this script uses a temporary cwd)
  (P5) dependencies: einops (repeat) · fvcore (import only) · wget (import only) → scratch shims (fvcore/wget are empty stubs, einops is the pip package)
  (P6) csm_triton/csms6s raise NameError in forward without the CUDA kernel → with --no-kernel, SelectiveScanOflex is replaced by the toolkit's pure-torch fallback (monkeypatch)

Goldens (model.pt):
- input: tests/golden/patchmix/features_input.pt (2,798,128) → torchvision Resize((1024,128)) (main.py:200-205) → expanded to batch 4 (4,1024,128)   [images]
- ckpt load: missing/unexpected keys with 'module.v.' stripped, head excluded, strict=False
- patch_embed output (2,96,32,256), blur kernel (K5 σ3), eval fp32 full forward: features/logits (blur on = SAR structure, blur off = FT structure),
  autocast fp16 forward (reference), CrossScanTriton vs torch CrossScan agreement
- 3 patch-mix types (time/freq/2d) as_code: applied to the patch_embed output after fixing the seeds (np+torch) → changed-axis indices, lam, index, checksum, and full-forward features
- patchmix_cl dual total loss (CE + L_time + L_freq; projector det_fill, τ 0.2) and each term, standalone PatchMixConLoss value
Random parameters: the head uses det_fill(linspace), the projector the default initialization after torch.manual_seed(0) (same module order as the toolkit Projector). The goldens are produced with the model in **eval** mode (DropPath disabled).
"""
import argparse
import json
import os
import pathlib
import sys
import types

import numpy as np
import torch
import torch.nn as nn


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="tests/golden/lungsrad")
    ap.add_argument("--input", default="tests/golden/patchmix/features_input.pt")
    ap.add_argument("--no-kernel", action="store_true", help="without the CUDA kernel, run the original code with the toolkit torch fallback as well")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = pathlib.Path.cwd()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()
    ckpt = pathlib.Path(args.ckpt).resolve()
    x_in = torch.load(root / args.input)
    sys.path.insert(0, str(repo))

    # (P5) stubs: fvcore / wget only need to be importable
    for name in ("fvcore", "fvcore.nn", "wget"):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                m = types.ModuleType(name)
                if name == "fvcore.nn":
                    for attr in ("FlopCountAnalysis", "flop_count_str", "flop_count", "parameter_count"):
                        setattr(m, attr, None)
                sys.modules[name] = m
    from models_DASS import csms6s  # noqa: E402
    kernel = hasattr(csms6s, "selective_scan_cuda_oflex")
    if args.no_kernel or not kernel:
        # (P6) the original code also uses the toolkit fallback
        from rsc_toolkit.models.backbone.vmamba.selective_scan import selective_scan_torch

        class _Fallback:
            @staticmethod
            def apply(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
                return selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)

        csms6s.SelectiveScanOflex = _Fallback
        import models_DASS.vmamba_patchmix as vp
        vp.SelectiveScanOflex = _Fallback
        kernel = False
    print("selective_scan CUDA kernel:", kernel, flush=True)

    from models_DASS.ast_models import DASS  # noqa: E402
    from models_DASS.csm_triton import CrossScanTriton, CrossMergeTriton  # noqa: E402
    from models_DASS.csms6s import CrossScan, CrossMerge  # noqa: E402
    # models/__init__.py and method/__init__.py pull in other backbones (timm 0.4 API etc.), so load per file (clone unmodified)
    import importlib.util

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    Projector = _load(repo / "models" / "projector.py", "lungsrad_projector").Projector
    PatchMixConLoss = _load(repo / "method" / "patchmix_cl.py", "lungsrad_patchmix_cl").PatchMixConLoss
    from torchvision import transforms  # noqa: E402

    dev = torch.device(args.device)
    # numerical reproducibility: disable cuDNN/matmul TF32 (the original uses the defaults = conv TF32 allowed; the goldens use the exact fp32 path so the toolkit's 1e-3 comparison is not affected by algorithm selection)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # (P4) relative ckpt path: temporary cwd
    cwd = pathlib.Path(os.environ.get("LUNGSRAD_GOLDEN_CWD", str(out / "_cwd")))
    (cwd / "pretrained_models").mkdir(parents=True, exist_ok=True)
    link = cwd / "pretrained_models" / "DASS_medium_v2.pth"
    if not link.exists():
        link.symlink_to(ckpt)
    os.chdir(cwd)

    # input: (2,798,128) → ToTensor layout (1,798,128) → Resize((1024,128))
    if x_in.dim() == 3:
        x_in = x_in.unsqueeze(1)
    assert tuple(x_in.shape[1:]) == (1, 798, 128), tuple(x_in.shape)
    resize = transforms.Resize(size=(1024, 128))
    images = torch.stack([resize(xi) for xi in x_in]).squeeze(1)  # (2,1024,128)
    # expand to batch 4 (the contrastive loss is trivial with B=2): original 2 + time flip + ×0.8 + fixed noise
    torch.manual_seed(1234)
    images = torch.cat([images, images[:1].flip(1), images[1:] * 0.8 + 0.05 * torch.randn_like(images[1:])], dim=0)  # (4,1024,128)
    labels = torch.tensor([0, 3, 1, 2])
    B = images.shape[0]

    def build(blur: bool, patch_mix: bool):
        torch.manual_seed(0)
        m = DASS(label_dim=4, imagenet_pretrain=False, audioset_pretrain=True, enable_patch_mix=patch_mix, mix_beta=1.0,
                 model_size="medium", blur_blocks=(set([2, 3]) if blur else set()), gaussian_blur=True, kernel_size=5, sigma=3.0,
                 blur_stage=set([2]), verbose=False)
        det_fill(m.v.classifier.head.weight)
        det_fill(m.v.classifier.head.bias)
        return m.to(dev).eval()

    # ckpt key comparison (re-runs the original load logic as-is)
    raw = torch.load(ckpt, map_location="cpu")
    mod = {k.replace("module.v.", ""): v for k, v in raw.items() if "classifier.head" not in k}
    tmp = build(True, True)
    res = tmp.v.load_state_dict(mod, strict=False)
    ckpt_info = {"n_keys_raw": len(raw), "n_keys_loaded": len(mod), "missing": sorted(res.missing_keys), "unexpected": sorted(res.unexpected_keys),
                 "patch_embed0_shape": list(raw["module.v.patch_embed.0.weight"].shape), "head_shape": list(raw["module.v.classifier.head.weight"].shape)}
    print("ckpt:", ckpt_info, flush=True)
    del tmp

    # input_raw: first 2 samples before Resize (2,798,128)
    golden = {"images": images.clone(), "labels": labels.clone(), "input_raw": x_in.squeeze(1).clone(), "ckpt": ckpt_info, "kernel": kernel}
    x4 = images.to(dev).unsqueeze(1).transpose(2, 3)  # (B,1,128,1024) as in DASS.forward
    golden["model_input_bchw"] = x4.cpu()

    with torch.no_grad():
        model = build(True, True)
        pe = model.v.patch_embed(x4)
        golden["patch_embed"] = pe.float().cpu()
        blur_blocks = [(i, j) for i, layer in enumerate(model.v.layers) for j, blk in enumerate(layer.blocks) if getattr(blk.op, "blur_enable", False)]
        golden["blur_blocks"] = blur_blocks
        g = model.v.layers[2].blocks[2].op._gaussian_kernel1d(5, 3.0, dev, torch.float32)
        golden["blur_kernel"] = g.cpu()
        # CrossScan: triton vs torch
        xs_t = CrossScanTriton.apply(pe.contiguous())
        xs_p = CrossScan.apply(pe.contiguous())
        ys = torch.randn(B, 4, pe.shape[1], pe.shape[2], pe.shape[3], device=dev)
        golden["crossscan_triton_vs_torch_maxdiff"] = float((xs_t - xs_p).abs().max())
        golden["crossmerge_triton_vs_torch_maxdiff"] = float((CrossMergeTriton.apply(ys) - CrossMerge.apply(ys)).abs().max())

        # full forward fp32 (blur on = SAR structure)
        feats_on, logits_on = model.v(x4)
        golden["features_blur"] = feats_on.float().cpu()
        golden["logits_blur"] = logits_on.float().cpu()
        # autocast fp16 (reference)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            f16, l16 = model(images.to(dev))
        golden["features_blur_fp16"] = f16.float().cpu()
        golden["logits_blur_fp16"] = l16.float().cpu()
        # intermediate: per-stage outputs
        h = pe
        stage_out = []
        for layer in model.v.layers:
            h = layer(h)
            stage_out.append(h.float().cpu())
        golden["stage_outputs_blur"] = {0: stage_out[0][:, :, :, :16].contiguous(), 3: stage_out[3]}   # save space: part of stage0 + the final stage

        # blur off (FT structure)
        model_off = build(False, False)
        assert not any(getattr(b.op, "blur_enable", False) for l in model_off.v.layers for b in l.blocks)
        feats_off, logits_off = model_off.v(x4)
        golden["features_noblur"] = feats_off.float().cpu()
        golden["logits_noblur"] = logits_off.float().cpu()
        del model_off

        # 3 patch-mix types (as_code) — fixed randomness: np.random.seed / torch.manual_seed right before each call
        mixes = {}
        for mix_type, seed in (("time", 11), ("freq", 22), ("2d", 33)):
            np.random.seed(seed)
            torch.manual_seed(seed)
            x_pe = model.v.patch_embed(x4)
            fn = {"time": model.v.patch_mix_time_only, "freq": model.v.patch_mix_freq_only, "2d": model.v.patch_mix_2d_ast_equiv}[mix_type]
            xm, y_a, y_b, lam, index = fn(x_pe, labels.to(dev))
            changed = (xm != pe)
            # keep the golden small: store only the changed-axis indices and the (2d) token mask instead of the mixed patch_embed tensor — the values are compared via mixed_features
            mixes[mix_type] = {
                "seed": seed, "lam": float(lam), "index": index.cpu(), "y_b": y_b.cpu(),
                "changed_dims": {"dim1": changed.any(dim=(0, 2, 3)).nonzero().flatten().cpu(), "dim2": changed.any(dim=(0, 1, 3)).nonzero().flatten().cpu(),
                                 "dim3": changed.any(dim=(0, 1, 2)).nonzero().flatten().cpu()},
                "changed_tokens_cf": changed.any(dim=(0, 3)).flatten().nonzero().flatten().cpu(),   # (channel×frequency) token index (2d as_code)
                "mixed_patch_embed_sum": float(xm.float().sum()), "mixed_patch_embed_checksum": xm.float().cpu().flatten()[::997].clone(),
            }
            np.random.seed(seed)
            torch.manual_seed(seed)
            mf, y_a2, y_b2, lam2, index2 = model.v(x4, y=labels.to(dev), patch_mix=True, mix_type=mix_type)
            assert abs(lam2 - lam) < 1e-12 and torch.equal(index2.cpu(), index.cpu())
            mixes[mix_type]["mixed_features"] = mf.float().cpu()
        golden["patchmix"] = mixes

        # patchmix_cl dual loss (main.py:310-351), projector det_fill
        torch.manual_seed(0)
        projector = Projector(768, 768).to(dev).eval()  # default init (seed 0) — det_fill makes the values large, so the loss degenerates to ln B after normalization
        crit = nn.CrossEntropyLoss()
        con = PatchMixConLoss(temperature=0.2)
        ns = types.SimpleNamespace(negative_pair="all")
        feats, output = model.v(x4)
        class_loss = crit(output, labels.to(dev))
        proj1 = projector(feats)
        np.random.seed(11); torch.manual_seed(11)
        mt, _, lbt, lam_t, idx_t = model.v(x4, y=labels.to(dev), patch_mix=True, mix_type="time")
        loss_time = con(proj1, projector(mt.detach().clone()), labels.to(dev), lbt, lam_t, idx_t, ns)
        np.random.seed(22); torch.manual_seed(22)
        mfq, _, lbf, lam_f, idx_f = model.v(x4, y=labels.to(dev), patch_mix=True, mix_type="freq")
        loss_freq = con(proj1, projector(mfq.detach().clone()), labels.to(dev), lbf, lam_f, idx_f, ns)
        total = class_loss + 1.0 * loss_time + 1.0 * loss_freq
        golden["loss"] = {"ce": float(class_loss), "time": float(loss_time), "freq": float(loss_freq), "total": float(total),
                          "lam_t": float(lam_t), "lam_f": float(lam_f), "index_t": idx_t.cpu(), "index_f": idx_f.cpu(),
                          "proj1": proj1.float().cpu()}
        # PatchMixConLoss standalone (fixed inputs)
        torch.manual_seed(5)
        p1, p2 = torch.randn(4, 768), torch.randn(4, 768)
        idx = torch.tensor([2, 3, 0, 1])
        golden["conloss_unit"] = {"p1": p1, "p2": p2, "index": idx, "lam": 0.37,
                                  "value": float(con(p1.to(dev), p2.to(dev), torch.tensor([0, 1, 2, 3]).to(dev), None, 0.37, idx.to(dev), ns))}
        golden["projector_state"] = {k: v.cpu() for k, v in projector.state_dict().items()}
        golden["n_params"] = sum(p.numel() for p in model.v.parameters())
        golden["state_keys"] = {k: list(v.shape) for k, v in model.v.state_dict().items()}

    tmp = out / "model.pt.tmp"
    torch.save(golden, tmp)
    tmp.replace(out / "model.pt")
    import timm, torchvision
    prov = {"repo": str(repo), "ckpt": str(ckpt), "torch": torch.__version__, "timm": timm.__version__, "torchvision": torchvision.__version__,
            "device": str(dev), "gpu": (torch.cuda.get_device_name(0) if dev.type == "cuda" else None), "selective_scan_kernel": kernel,
            "patches": ["P1..P6 in docstring (clone unmodified; P3 blur_blocks=∅, P4 symlink cwd, P5 shims, P6 fallback only if --no-kernel)"]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print("saved", out / "model.pt", flush=True)
    print({k: (v if not torch.is_tensor(v) else tuple(v.shape)) for k, v in golden.items() if k in ("ckpt", "kernel", "blur_blocks", "crossscan_triton_vs_torch_maxdiff", "crossmerge_triton_vs_torch_maxdiff", "loss", "n_params")}, flush=True)


if __name__ == "__main__":
    main()
