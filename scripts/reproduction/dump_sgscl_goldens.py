"""
Produces official-code parity goldens for the 4 SFT backbones (ResNet18/34, EfficientNet-B0, CNN6) of the SG-SCL repository.

Run (reference environment — torch 1.13 + torchvision 0.14 + timm 0.4.5):
  .venv-ref-timm045/bin/python scripts/reproduction/dump_sgscl_goldens.py \
      --repo <sgscl clone> --out tests/golden/sgscl_sft

Notes:
- The 1-channel first conv of ResNet/EfficientNet stays randomly initialized in the original code too
  (it is removed from the ImageNet sd). To make values comparable, both the dump and the test overwrite it
  with the same deterministic pattern (linspace) before running forward.
- CNN6 needs the PANN ckpt at ./pretrained_models/Cnn6_mAP=0.343.pth (relative to cwd)
  — the repository archive copy is symlinked there before running.
"""
import argparse
import json
import os
import pathlib
import sys

import torch

# The original CNN6.load_sl_official_weights calls torch.load without map_location.
# The PANN ckpt was saved on CUDA and fails in the CPU-only reference environment, so force CPU
# (only the deserialization location changes; the weight values are identical).
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})


def det_fill(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="tests/golden/sgscl_sft")
    args = ap.parse_args()

    repo_root = pathlib.Path.cwd()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # link the archive copy to where CNN6 looks for it
    link = repo_root / "pretrained_models" / "Cnn6_mAP=0.343.pth"
    target = repo_root / "pretrained_models" / "upstream" / "Cnn6_mAP=0.343.pth"
    if not link.exists() and target.exists():
        os.symlink(target, link)

    sys.path.insert(0, str(pathlib.Path(args.repo).resolve()))
    from models.resnet import ResNet18, ResNet34  # noqa: E402
    from models.efficientnet import EfficientNet_B0  # noqa: E402
    from models.cnn import CNN6  # noqa: E402

    import torchvision, timm  # noqa: E402
    json.dump({"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__},
              open(out / "reference_versions.json", "w"), indent=1)

    torch.manual_seed(123)
    x = torch.randn(2, 1, 798, 128)
    torch.save(x, out / "features_input.pt")

    specs = {}
    for name, ctor, det_convs in [
        ("resnet18", ResNet18, lambda m: [m.conv1.weight]),
        ("resnet34", ResNet34, lambda m: [m.conv1.weight]),
        ("efficientnet_b0", EfficientNet_B0, lambda m: [m.features[0][0].weight]),
        ("cnn6", CNN6, lambda m: []),
    ]:
        m = ctor()
        m.load_sl_official_weights()
        for t in det_convs(m):
            det_fill(t)
        m.eval()
        with torch.no_grad():
            feats = m(x)
        torch.save(feats.float(), out / f"{name}_features.pt")
        specs[name] = {k: list(v.shape) for k, v in m.state_dict().items()}
        print(f"{name:16} feats {tuple(feats.shape)} mean {float(feats.mean()):+.5f}")

    json.dump(specs, open(out / "state_dict_shapes.json", "w"), indent=0, sort_keys=True)
    print("done:", out)


if __name__ == "__main__":
    main()
