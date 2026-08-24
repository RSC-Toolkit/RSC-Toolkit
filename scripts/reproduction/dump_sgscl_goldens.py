"""
SG-SCL 저장소의 SFT 백본 4종(ResNet18/34, EfficientNet-B0, CNN6)에서 P0 골든을 뜬다.

실행 (참조 환경 — torch 1.13 + torchvision 0.14 + timm 0.4.5):
  .venv-ref-timm045/bin/python scripts/reproduction/dump_sgscl_goldens.py \
      --repo <sgscl 클론> --out tests/golden/sgscl_sft

주의:
- ResNet/EfficientNet 의 1채널 첫 conv 는 원본에서도 랜덤 초기화로 남는다
  (ImageNet sd 에서 삭제됨). 값 비교가 가능하도록 덤프/테스트 양쪽에서 같은
  결정 패턴(linspace)으로 덮은 뒤 forward 를 뜬다.
- CNN6 는 PANN ckpt 가 ./pretrained_models/Cnn6_mAP=0.343.pth (cwd 기준) 에
  있어야 한다 — 저장소 아카이브를 심볼릭 링크로 걸어 실행한다.
"""
import argparse
import json
import os
import pathlib
import sys

import torch

# 원본 CNN6.load_sl_official_weights 는 map_location 없이 torch.load 를 부른다.
# PANN ckpt 는 CUDA 저장이라 CPU 전용 참조 환경에서는 실패하므로 CPU 로 강제한다
# (역직렬화 위치만 바뀔 뿐 가중치 값은 동일).
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

    # CNN6 이 찾는 위치로 아카이브 링크
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
    print("완료:", out)


if __name__ == "__main__":
    main()
