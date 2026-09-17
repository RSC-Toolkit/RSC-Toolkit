"""
Produces official-code parity goldens for the SG-SCL method (model, loss, two-view augmentation) from the official repository code.

It has two parts (with different run environments):

  ref  — AST(audioset) state_dict/forward + MetaCL + Projector composite loss.
         Runs in the reference environment (torch 1.13 + timm 0.4.5):
           .venv-ref-timm045/bin/python scripts/reproduction/dump_sgscl_method_goldens.py \
               ref --repo <sgscl clone> --out tests/golden/sgscl_method

  pipeline — batch-level SpecAugment two-view (cumulative masking) goldens. The original util/augmentation.py
         imports nlpaug, so a stub is inserted and it runs **in our venv**
         (RNG consumption is only numpy/python/torch indexing, so it is version-independent; same approach as the
         patch-mix pipeline goldens):
           .venv/bin/python scripts/reproduction/dump_sgscl_method_goldens.py \
               pipeline --repo <sgscl clone> --out tests/golden/sgscl_method

Notes:
- MetaCL (method/mcl.py) hard-codes .cuda(), so it is patched to identity on CPU.
- AST looks for ./pretrained_models/audioset_10_10_0.4593.pth (relative to cwd) —
  the archive copy (pretrained_models/upstream/audioset_0.4593.pth) is symlinked there.
- Projector parameters are overwritten with det_fill(linspace) and it runs in eval mode (default BN running statistics)
  — with the same conditions on both sides, comparing the composition (projection→MetaCL) is sufficient.
"""
import argparse
import json
import os
import pathlib
import random
import sys
import types

import numpy as np
import torch

# for the CPU reference environment: .cuda() → identity, torch.load → forced CPU
torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})


def det_fill(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def dump_ref(repo: pathlib.Path, out: pathlib.Path) -> None:
    # AST audioset ckpt link (the original hard-codes a cwd-relative path)
    link = pathlib.Path("pretrained_models/audioset_10_10_0.4593.pth")
    target = pathlib.Path("pretrained_models/upstream/audioset_0.4593.pth")
    if not link.exists() and target.exists():
        os.symlink(target.resolve(), link)

    sys.path.insert(0, str(repo))
    from models.ast import ASTModel  # noqa: E402
    from models.projector import Projector  # noqa: E402
    from method.mcl import MetaCL  # noqa: E402

    import timm  # noqa: E402
    json.dump({"torch": torch.__version__, "timm": timm.__version__},
              open(out / "reference_versions.json", "w"), indent=1)

    # ---- 1) SG-SCL AST: script settings as-is (including the swapped h/w quirk) ----
    m = ASTModel(label_dim=4, fstride=10, tstride=10, input_fdim=798, input_tdim=128,
                 imagenet_pretrain=True, audioset_pretrain=True, model_size="base384",
                 verbose=False, domain_label_dim=527)
    m.eval()
    torch.manual_seed(123)
    x = torch.randn(2, 1, 798, 128)
    torch.save(x, out / "features_input.pt")
    with torch.no_grad():
        feats = m(x, training=False)
    torch.save(feats.float(), out / "ast_features.pt")
    json.dump({k: list(v.shape) for k, v in m.state_dict().items()},
              open(out / "ast_state_dict_shapes.json", "w"), indent=0, sort_keys=True)
    print(f"ast feats {tuple(feats.shape)} mean {float(feats.mean()):+.6f}")

    # ---- 2) MetaCL loss (with/without meta) ----
    torch.manual_seed(202)
    proj1 = torch.randn(6, 16)
    proj2 = torch.randn(6, 16)
    meta = torch.tensor([0, 1, 0, 3, 1, 0])
    mcl = MetaCL(temperature=0.06)
    loss_meta = mcl(proj1, proj2, meta)
    loss_none = mcl(proj1, proj2, None)
    torch.save({"proj1": proj1, "proj2": proj2, "meta": meta,
                "loss_meta": loss_meta.detach(), "loss_none": loss_none.detach()},
               out / "metacl_golden.pt")
    print(f"metacl loss_meta {float(loss_meta):.6f} loss_none {float(loss_none):.6f}")

    # ---- 3) composition: Projector + target_type='project1_project2block' + MetaCL ----
    proj = Projector(16, 16)
    for _, p in sorted(proj.named_parameters()):
        det_fill(p)
    proj.eval()
    torch.manual_seed(303)
    feat1 = torch.randn(6, 16)
    feat2 = torch.randn(6, 16)
    from copy import deepcopy
    p1 = proj(feat1)
    p2 = deepcopy(proj(feat2).detach())  # original main.py:409-411
    comp_loss = mcl(p1, p2, meta)
    torch.save({"feat1": feat1, "feat2": feat2, "meta": meta,
                "p1": p1.detach(), "p2": p2, "loss": comp_loss.detach()},
               out / "sgscl_composition_golden.pt")
    print(f"composition loss {float(comp_loss):.6f}")


def dump_pipeline(repo: pathlib.Path, out: pathlib.Path) -> None:
    # stubs for heavy/unneeded imports in the original util/augmentation.py (not used on the masking path)
    nlpaug = types.ModuleType("nlpaug")
    nlpaug_a = types.ModuleType("nlpaug.augmenter")
    nlpaug_aa = types.ModuleType("nlpaug.augmenter.audio")
    sys.modules.setdefault("nlpaug", nlpaug)
    sys.modules.setdefault("nlpaug.augmenter", nlpaug_a)
    sys.modules.setdefault("nlpaug.augmenter.audio", nlpaug_aa)

    sys.path.insert(0, str(repo))
    from util.augmentation import SpecAugment  # noqa: E402

    args = types.SimpleNamespace(specaug_policy="icbhi_ast_sup", specaug_mask="mean")
    transforms = SpecAugment(args)

    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    x = torch.randn(2, 1, 798, 128)  # [B, 1, time, freq] — original train batch
    torch.save(x.clone(), out / "twoview_input.pt")

    # original train(): transforms applied twice to the same batch tensor — in-place, so the masks accumulate
    images = x.clone()
    v1 = transforms(images).clone()
    v2 = transforms(images).clone()
    torch.save({"v1": v1, "v2": v2}, out / "twoview_golden.pt")
    print(f"twoview fraction of elements with v1==v2 {float((v1 == v2).float().mean()):.4f} "
          f"(with cumulative masking below 1.0, while most v1 mask positions are kept)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["ref", "pipeline"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="tests/golden/sgscl_method")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()

    if args.part == "ref":
        dump_ref(repo, out)
    else:
        dump_pipeline(repo, out)
    print("done:", out)


if __name__ == "__main__":
    main()
