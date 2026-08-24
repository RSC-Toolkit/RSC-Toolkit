"""
Patch-Mix 공식 코드(timm 0.4.5)에서 P0 골든 파일을 뜬다.

실행 (참조 환경 필수 — 툴킷 venv 가 아니라 구버전 환경):
  .venv-ref-timm045/bin/python scripts/reproduction/dump_patchmix_goldens.py \
      --repo <patch-mix 저장소 클론 경로> --ckpt pretrained_models/upstream/audioset_0.4593.pth

산출물 (tests/golden/patchmix/):
  - reference_versions.json  : 참조 환경의 torch/timm 버전
  - state_dict_shapes.json   : 참조 ASTModel(audioset, 798/128)의 {키: 모양}
  - features_input.pt / features_golden.pt : 고정 입력과 참조 forward 출력 (B=2, 768)
  - loss_goldens.pt          : 손실 입력 텐서 + 참조 손실 값

주의:
- 참조 PatchMixConLoss 는 .cuda() 를 하드코딩하므로 CPU 실행을 위해
  Tensor.cuda / Module.cuda 를 항등으로 패치한다 (수학에는 영향 없음).
- 입력 텐서를 함께 저장하므로 버전 간 RNG 차이는 결과에 영향을 주지 않는다.
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

# CPU 전용 환경에서 참조 코드의 하드코딩된 .cuda() 를 무력화
torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
nn.Module.cuda = lambda self, *a, **k: self  # type: ignore[assignment]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="patch-mix 공식 저장소 클론 경로")
    ap.add_argument("--ckpt", required=True, help="audioset_0.4593.pth 경로")
    ap.add_argument("--out", default="tests/golden/patchmix")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    ckpt = pathlib.Path(args.ckpt).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import timm
    json.dump(
        {"torch": torch.__version__, "timm": timm.__version__, "repo": str(repo)},
        open(out / "reference_versions.json", "w"), indent=1,
    )

    sys.path.insert(0, str(repo))

    # 참조 ast.py 는 './pretrained_models/audioset_10_10_0.4593.pth' 를 cwd 기준으로 찾는다
    workdir = tempfile.mkdtemp(prefix="patchmix_golden_")
    os.makedirs(os.path.join(workdir, "pretrained_models"), exist_ok=True)
    os.symlink(ckpt, os.path.join(workdir, "pretrained_models", "audioset_10_10_0.4593.pth"))
    os.chdir(workdir)

    from models.ast import ASTModel  # noqa: E402  (참조 저장소)
    from method.patchmix import PatchMixLoss  # noqa: E402
    from method.patchmix_cl import PatchMixConLoss  # noqa: E402

    # ---- 모델 골든: main.py 의 AST 호출과 동일 (798/128 스왑 포함) ----
    torch.manual_seed(0)
    model = ASTModel(
        label_dim=4, fstride=10, tstride=10, input_fdim=798, input_tdim=128,
        imagenet_pretrain=True, audioset_pretrain=True, model_size="base384",
        verbose=False, mix_beta=1.0,
    )
    model.eval()

    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    json.dump(shapes, open(out / "state_dict_shapes.json", "w"), indent=0, sort_keys=True)

    torch.manual_seed(123)
    x = torch.randn(2, 1, 798, 128)
    with torch.no_grad():
        feats = model(x)
    torch.save(x, out / "features_input.pt")
    torch.save(feats.float(), out / "features_golden.pt")
    print("features golden:", tuple(feats.shape), "mean", float(feats.mean()))

    # ---- 손실 골든 (입력도 함께 저장) ----
    torch.manual_seed(7)
    g = {}
    g["pred"] = torch.randn(8, 4)
    g["y_a"] = torch.randint(0, 4, (8,))
    g["y_b"] = torch.randint(0, 4, (8,))
    g["lam"] = 0.37
    g["class_nums"] = [2063.0, 1215.0, 501.0, 363.0]
    weights = torch.tensor(g["class_nums"], dtype=torch.float32)
    weights = 1.0 / (weights / weights.sum())
    weights /= weights.sum()

    g["patchmix_loss_unweighted"] = float(PatchMixLoss(nn.CrossEntropyLoss())(g["pred"], g["y_a"], g["y_b"], g["lam"]))
    g["patchmix_loss_weighted"] = float(
        PatchMixLoss(nn.CrossEntropyLoss(weight=weights))(g["pred"], g["y_a"], g["y_b"], g["lam"])
    )

    g["proj1"] = torch.randn(8, 768)
    g["proj2"] = torch.randn(8, 768)
    g["index"] = torch.randperm(8)
    for np_mode in ("all", "diff_label"):
        loss = PatchMixConLoss(temperature=0.06)(
            g["proj1"], g["proj2"], g["y_a"], g["y_b"], g["lam"], g["index"],
            SimpleNamespace(negative_pair=np_mode),
        )
        g[f"patchmix_con_loss_{np_mode}"] = float(loss)

    torch.save(g, out / "loss_goldens.pt")
    print("loss goldens:", {k: v for k, v in g.items() if isinstance(v, float)})
    print("완료:", out)


if __name__ == "__main__":
    main()
