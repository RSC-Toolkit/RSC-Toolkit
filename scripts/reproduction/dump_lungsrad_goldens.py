"""
Lung-SRAD(RSC-Toolkit/Lung-SRAD d28d783) P0 골든 — 공식 models_DASS/{ast_models,vmamba_patchmix,csms6s,csm_triton}.py +
method/patchmix_cl.py + models/projector.py 를 **실 ckpt(DASS_medium_v2.pth)** 로 GPU 에서 실행해 저장한다.

  # 커널(선택): outputs/reference/dass_upstream/kernels/selective_scan 을 scratch 에 빌드 (README 절차, .venv 에 설치하지 않음)
  #   cp -r kernels/selective_scan $S/ss_src && (setup.py 의 wheel import 제거) && MAX_JOBS=6 .venv/bin/python setup.py build_ext --build-lib $S/pylib
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$S/pylib:$S/shims .venv/bin/python scripts/reproduction/dump_lungsrad_goldens.py \
      --repo outputs/reference/lungsrad --ckpt pretrained_models/upstream/dass/DASS_medium_v2.pth --out tests/golden/lungsrad [--no-kernel]

공개 코드는 그대로 실행되지 않는다. 이 스크립트는 main.py 를 import 하지 않고 모델·손실 모듈만 쓰므로 클론을 **수정하지 않는다**.
main.py 로 학습까지 돌리려면 필요한 최소 패치 목록(문서화, 본 골든에는 불필요):
  (P1) main.py:27   `from method import PatchMixLoss, PatchMixConLoss` → `from method.patchmix_cl import PatchMixConLoss` (PatchMixLoss 는 저장소에 없음)
  (P2) util/icbhi_util.py:168  `args.meta_mode` → `getattr(args, "meta_mode", None)` (argparse 미정의)
  (P3) FT 기준선(--gaussian_blur 없음): ast_models.py:278-283 이 고르는 plain vmamba.VSSM 은 forward(x) 만 받아 DASS.forward 의 kwargs 로 TypeError
       → DASS(gaussian_blur=True, blur_blocks=set()) 로 vmamba_patchmix.VSSM 을 blur 블록 ∅ 로 경유 (구조·가중치 동일). 본 스크립트도 이 방식.
  (P4) ckpt 경로 './pretrained_models/DASS_medium_v2.pth' 상대 하드코딩 → cwd 에 symlink (본 스크립트는 임시 cwd 로 처리)
  (P5) 의존성: einops(repeat) · fvcore(import 만) · wget(import 만) → scratch shims (fvcore/wget 은 빈 스텁, einops 는 pip 패키지)
  (P6) csm_triton/csms6s 는 CUDA 커널이 없으면 forward 에서 NameError → --no-kernel 이면 SelectiveScanOflex 를 툴킷 순수 torch 폴백으로 대체(monkeypatch)

골든 (model.pt):
- 입력: tests/golden/patchmix/features_input.pt (2,798,128) → torchvision Resize((1024,128)) (main.py:200-205) → 배치 4 로 확장 (4,1024,128)   [images]
- ckpt 로드: 'module.v.' 제거·head 제외·strict=False 의 missing/unexpected 키
- patch_embed 출력 (2,96,32,256), blur 커널(K5 σ3), eval fp32 전체 forward: features/logits (blur on = SAR 구조, blur off = FT 구조),
  autocast fp16 forward(참고), CrossScanTriton vs torch CrossScan 일치
- patch-mix 3종(time/freq/2d) as_code: seed 고정(np+torch) 후 patch_embed 출력에 적용 → 변경 축 index·lam·index·체크섬, 그리고 전체 forward 특징
- patchmix_cl dual 총손실(CE + L_time + L_freq; projector det_fill, τ 0.2) 및 각 항, PatchMixConLoss 단독 값
난수 파라미터: head 는 det_fill(linspace), projector 는 torch.manual_seed(0) 기본 초기화(툴킷 Projector 와 같은 모듈 순서). 골든은 모델을 **eval** 모드(DropPath 무효) 로 두고 만든다.
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
    ap.add_argument("--no-kernel", action="store_true", help="CUDA 커널 없이 원 코드도 툴킷 torch 폴백으로")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = pathlib.Path.cwd()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()
    ckpt = pathlib.Path(args.ckpt).resolve()
    x_in = torch.load(root / args.input)
    sys.path.insert(0, str(repo))

    # (P5) 스텁: fvcore / wget 은 import 만 된다
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
        # (P6) 원 코드도 툴킷 폴백으로
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
    # models/__init__.py, method/__init__.py 는 다른 백본(timm 0.4 API 등)을 끌어오므로 파일 단위로 로드 (클론 무수정)
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
    # 수치 재현성: cuDNN/matmul TF32 끔 (원본은 기본값 = conv TF32 허용; 골든은 fp32 정확 경로로 만들어 툴킷 1e-3 대조가 알고리즘 선택에 흔들리지 않게)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # (P4) ckpt 상대 경로: 임시 cwd
    cwd = pathlib.Path(os.environ.get("LUNGSRAD_GOLDEN_CWD", str(out / "_cwd")))
    (cwd / "pretrained_models").mkdir(parents=True, exist_ok=True)
    link = cwd / "pretrained_models" / "DASS_medium_v2.pth"
    if not link.exists():
        link.symlink_to(ckpt)
    os.chdir(cwd)

    # 입력: (2,798,128) → ToTensor 레이아웃 (1,798,128) → Resize((1024,128))
    if x_in.dim() == 3:
        x_in = x_in.unsqueeze(1)
    assert tuple(x_in.shape[1:]) == (1, 798, 128), tuple(x_in.shape)
    resize = transforms.Resize(size=(1024, 128))
    images = torch.stack([resize(xi) for xi in x_in]).squeeze(1)  # (2,1024,128)
    # 배치 4 로 확장(대조 손실이 B=2 에선 자명해짐): 원본 2 + 시간 뒤집기 + 0.8 배 + 고정 잡음
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

    # ckpt 키 대조 (원본 로드 로직 그대로 재실행)
    raw = torch.load(ckpt, map_location="cpu")
    mod = {k.replace("module.v.", ""): v for k, v in raw.items() if "classifier.head" not in k}
    tmp = build(True, True)
    res = tmp.v.load_state_dict(mod, strict=False)
    ckpt_info = {"n_keys_raw": len(raw), "n_keys_loaded": len(mod), "missing": sorted(res.missing_keys), "unexpected": sorted(res.unexpected_keys),
                 "patch_embed0_shape": list(raw["module.v.patch_embed.0.weight"].shape), "head_shape": list(raw["module.v.classifier.head.weight"].shape)}
    print("ckpt:", ckpt_info, flush=True)
    del tmp

    # input_raw: 앞 2 샘플의 Resize 전 (2,798,128)
    golden = {"images": images.clone(), "labels": labels.clone(), "input_raw": x_in.squeeze(1).clone(), "ckpt": ckpt_info, "kernel": kernel}
    x4 = images.to(dev).unsqueeze(1).transpose(2, 3)  # DASS.forward 의 (B,1,128,1024)
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

        # 전체 forward fp32 (blur on = SAR 구조)
        feats_on, logits_on = model.v(x4)
        golden["features_blur"] = feats_on.float().cpu()
        golden["logits_blur"] = logits_on.float().cpu()
        # autocast fp16 (참고)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            f16, l16 = model(images.to(dev))
        golden["features_blur_fp16"] = f16.float().cpu()
        golden["logits_blur_fp16"] = l16.float().cpu()
        # 중간: stage 별 출력
        h = pe
        stage_out = []
        for layer in model.v.layers:
            h = layer(h)
            stage_out.append(h.float().cpu())
        golden["stage_outputs_blur"] = {0: stage_out[0][:, :, :, :16].contiguous(), 3: stage_out[3]}   # 크기 절약: stage0 일부 + 최종 stage

        # blur off (FT 구조)
        model_off = build(False, False)
        assert not any(getattr(b.op, "blur_enable", False) for l in model_off.v.layers for b in l.blocks)
        feats_off, logits_off = model_off.v(x4)
        golden["features_noblur"] = feats_off.float().cpu()
        golden["logits_noblur"] = logits_off.float().cpu()
        del model_off

        # patch-mix 3종 (as_code) — 난수 고정: np.random.seed / torch.manual_seed 를 각 호출 직전에
        mixes = {}
        for mix_type, seed in (("time", 11), ("freq", 22), ("2d", 33)):
            np.random.seed(seed)
            torch.manual_seed(seed)
            x_pe = model.v.patch_embed(x4)
            fn = {"time": model.v.patch_mix_time_only, "freq": model.v.patch_mix_freq_only, "2d": model.v.patch_mix_2d_ast_equiv}[mix_type]
            xm, y_a, y_b, lam, index = fn(x_pe, labels.to(dev))
            changed = (xm != pe)
            # 골든 크기 절약: 믹스된 patch_embed 텐서 대신 변경 축 index·(2d) 토큰 mask 만 저장 — 값 자체는 mixed_features 로 대조
            mixes[mix_type] = {
                "seed": seed, "lam": float(lam), "index": index.cpu(), "y_b": y_b.cpu(),
                "changed_dims": {"dim1": changed.any(dim=(0, 2, 3)).nonzero().flatten().cpu(), "dim2": changed.any(dim=(0, 1, 3)).nonzero().flatten().cpu(),
                                 "dim3": changed.any(dim=(0, 1, 2)).nonzero().flatten().cpu()},
                "changed_tokens_cf": changed.any(dim=(0, 3)).flatten().nonzero().flatten().cpu(),   # (채널×주파수) 토큰 index (2d as_code)
                "mixed_patch_embed_sum": float(xm.float().sum()), "mixed_patch_embed_checksum": xm.float().cpu().flatten()[::997].clone(),
            }
            np.random.seed(seed)
            torch.manual_seed(seed)
            mf, y_a2, y_b2, lam2, index2 = model.v(x4, y=labels.to(dev), patch_mix=True, mix_type=mix_type)
            assert abs(lam2 - lam) < 1e-12 and torch.equal(index2.cpu(), index.cpu())
            mixes[mix_type]["mixed_features"] = mf.float().cpu()
        golden["patchmix"] = mixes

        # patchmix_cl dual 손실 (main.py:310-351), projector det_fill
        torch.manual_seed(0)
        projector = Projector(768, 768).to(dev).eval()  # 기본 초기화(seed 0) — det_fill 은 값이 커져 정규화 후 손실이 ln B 로 퇴화
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
        # PatchMixConLoss 단독 (고정 입력)
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
            "patches": ["P1..P6 in docstring (클론 무수정; P3 blur_blocks=∅, P4 symlink cwd, P5 shims, P6 fallback only if --no-kernel)"]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print("saved", out / "model.pt", flush=True)
    print({k: (v if not torch.is_tensor(v) else tuple(v.shape)) for k, v in golden.items() if k in ("ckpt", "kernel", "blur_blocks", "crossscan_triton_vs_torch_maxdiff", "crossmerge_triton_vs_torch_maxdiff", "loss", "n_params")}, flush=True)


if __name__ == "__main__":
    main()
