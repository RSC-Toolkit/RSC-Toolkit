"""
Official-code parity goldens for the per-view AST of MVST — runs the official {view}/models/ast.py in the reference environment (timm 0.4.5).
  .venv-ref-timm045/bin/python scripts/reproduction/dump_mvst_goldens.py --repo <MVST clone> --out tests/golden/mvst
The audioset branch reads ./pretrained_models/audioset_16_16_0.4422.pth, so symlink the converted weights there.
Parts that remain random (pos_embed for all views, patch proj for non-16 views, mlp_head) are overwritten with det_fill on both sides before comparing the forward pass.
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import types

import torch

torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="tests/golden/mvst")
    ap.add_argument("--views", nargs="*", type=int, default=[16, 32, 64, 128, 256])
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name in ("wget",):
        sys.modules.setdefault(name, types.ModuleType(name))
    torch.manual_seed(123)
    x = torch.randn(2, 1, 1024, 256)
    torch.save(x, out / "features_input.pt")
    shapes = {}
    for v in args.views:
        spec = importlib.util.spec_from_file_location(f"ast_{v}", pathlib.Path(args.repo) / str(v) / "models" / "ast.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        m = mod.ASTModel(label_dim=4, imagenet_pretrain=True, audioset_pretrain=True, verbose=False)
        det_fill(m.v.pos_embed); det_fill(m.v.patch_embed.proj.weight); det_fill(m.v.patch_embed.proj.bias)
        m.eval()
        with torch.no_grad():
            feats = m(x)
        torch.save(feats.float(), out / f"view{v}_features.pt")
        shapes[str(v)] = {k: list(t.shape) for k, t in m.state_dict().items()}
        print(f"view {v}: feats {tuple(feats.shape)} mean {float(feats.mean()):+.5f} keys {len(shapes[str(v)])}")
    json.dump(shapes, open(out / "state_dict_shapes.json", "w"), indent=0, sort_keys=True)
    print("done:", out)


if __name__ == "__main__":
    main()
