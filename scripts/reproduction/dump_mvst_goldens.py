"""
MVST 뷰별 AST 의 P0 골든 — 공식 {view}/models/ast.py 를 참조 환경(timm 0.4.5)에서 실행.
  .venv-ref-timm045/bin/python scripts/reproduction/dump_mvst_goldens.py --repo <MVST 클론> --out tests/golden/mvst
audioset 분기가 ./pretrained_models/audioset_16_16_0.4422.pth 를 읽으므로 변환본을 심볼릭 링크로 걸어 둘 것.
랜덤으로 남는 부분(pos_embed 전 뷰, patch proj 비-16 뷰, mlp_head)은 양쪽에서 det_fill 로 덮어 forward 를 대조한다.
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
    print("완료:", out)


if __name__ == "__main__":
    main()
