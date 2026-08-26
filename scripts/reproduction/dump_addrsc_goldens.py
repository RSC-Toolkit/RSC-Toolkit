"""
ADD-RSC(deegy666/ADD-RSC e2b0f82) P0 골든 — 공식 models/{adapt_diff_denoise,ast,bias_denoise_loss}.py 를 timm 0.4.5 참조 venv 에서 실행.
(데이터 파이프라인은 patch-mix 와 동일해 tests/golden/patchmix* 골든으로 이미 대조됨; 여기서는 모델·손실만.)

  CUDA_VISIBLE_DEVICES= .venv-ref-timm045/bin/python scripts/reproduction/dump_addrsc_goldens.py \
      --repo outputs/reference/addrsc --ckpt pretrained_models/upstream/audioset_0.4593.pth --out tests/golden/addrsc

골든 (model.pt):
- 입력: tests/golden/patchmix/features_input.pt (2, 798, 128) fbank → transforms.Resize((1024,256)) (원본 main.py:176-181) → (2,1024,256)
- 공식 DiffTransformerLayer(256,8,6) + ASTModel(fstride=tstride=16, input_fdim=1024, input_tdim=256, audioset_pretrain, ckpt) + classifier=deepcopy(mlp_head)
  난수 초기화 파라미터(잡음제거 층 전체, classifier, v.pos_embed) 는 det_fill(linspace) 로 결정화 → eval fp32 forward:
  denoise_images, aux_logits, features, output + LabelSmoothing/CE/총 손실. AudioSet 로드 결과(로드 키·모양 불일치 키), state_dict 키/모양.
"""
import argparse
import json
import pathlib
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="tests/golden/addrsc")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(args.repo).resolve()))
    # 참조 venv 의 torch(cu130) 는 이 서버 드라이버로 CUDA 초기화가 안 된다 → 공식 코드의 cuda 분기(DataParallel/map_location) 를 CPU 로 강제
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    class _CPUDataParallel(nn.Module):  # DataParallel 대체: state_dict 키 'module.*' 접두어와 .module 만 재현 (GPU 없이)
        def __init__(self, module, *a, **k):
            super().__init__()
            self.module = module

        def forward(self, *a, **k):
            return self.module(*a, **k)

    torch.nn.DataParallel = _CPUDataParallel
    # 원본 차분 어텐션 forward 의 죽은 코드 `nn.LayerNorm(N).to('cuda:0')` (결과 미사용) — GPU 없이 돌리기 위해 그 .to 만 무시
    _orig_to = nn.Module.to

    def _to(self, *a, **k):
        if a and a[0] == "cuda:0":
            return self
        return _orig_to(self, *a, **k)

    nn.Module.to = _to
    import timm
    from torchvision import transforms
    from models.adapt_diff_denoise import DiffTransformerLayer
    from models.ast import ASTModel
    from models.bias_denoise_loss import LabelSmoothingLoss

    x = torch.load("tests/golden/patchmix/features_input.pt")
    if x.dim() == 3:
        x = x.unsqueeze(1)  # (2,1,798,128)
    assert tuple(x.shape[1:]) == (1, 798, 128), tuple(x.shape)
    resize = transforms.Resize(size=(1024, 256))
    images = torch.stack([resize(xi) for xi in x])  # (2,1,1024,256)
    labels = torch.tensor([0, 3])

    torch.manual_seed(0)
    denoise = DiffTransformerLayer(d_model=256, num_heads=8, depth=6)
    model = ASTModel(input_fdim=1024, input_tdim=256, label_dim=4, audioset_pretrain=True, pretrained_path=args.ckpt)
    classifier = deepcopy(model.mlp_head)
    # 난수 파라미터 결정화
    for p in denoise.parameters():
        det_fill(p)
    for p in classifier.parameters():
        det_fill(p)
    det_fill(model.v.pos_embed)

    denoise.eval(); model.eval(); classifier.eval()
    crit_ce, crit_ls = nn.CrossEntropyLoss(), LabelSmoothingLoss()
    with torch.no_grad():
        imgs = images.squeeze(1)
        denoise_images, denoise_feature = denoise(imgs)
        features = model(denoise_images.unsqueeze(1))
        output = classifier(features)
        denoise_loss = crit_ls(denoise_feature, labels)
        class_loss = crit_ce(output, labels)
        loss = 0.5 * denoise_loss + 0.5 * class_loss

    # AudioSet 로드 결과 재현 기록 (ast.py 130-137 로직을 같은 ckpt 로 다시 돌려 기록)
    weights = torch.load(args.ckpt, map_location="cpu")
    skeleton = ASTModel(input_fdim=1024, input_tdim=256, label_dim=527, audioset_pretrain=False, verbose=False)
    cur = torch.nn.DataParallel(skeleton).state_dict()
    loaded = sorted(k for k in cur if k in weights and cur[k].size() == weights[k].size())
    mismatch = sorted(k for k in cur if k in weights and cur[k].size() != weights[k].size())

    torch.save(
        {
            "input": x, "images": images, "labels": labels,
            "denoise_images": denoise_images, "aux_logits": denoise_feature, "features": features, "output": output,
            "denoise_loss": denoise_loss, "class_loss": class_loss, "loss": loss,
            "denoise_keys": {k: list(v.shape) for k, v in denoise.state_dict().items()},
            "v_keys": {k: list(v.shape) for k, v in model.v.state_dict().items()},
            "classifier_keys": {k: list(v.shape) for k, v in classifier.state_dict().items()},
            "lambda_init": denoise.attn.lambda_init,
            "audioset_loaded": loaded, "audioset_mismatch": mismatch,
            "pos_embed_shape": list(model.v.pos_embed.shape),
        },
        out / "model.pt",
    )
    json.dump({"repo": str(pathlib.Path(args.repo).resolve()), "ckpt": args.ckpt, "torch": torch.__version__, "timm": timm.__version__},
              open(out / "provenance.json", "w"), indent=2)
    print("output", output, "aux", denoise_feature, "loss", float(loss), "mismatch", mismatch, "n_loaded", len(loaded))


if __name__ == "__main__":
    main()
