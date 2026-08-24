"""
SG-SCL 방법(모델·손실·두 뷰 증강)의 P0 골든을 공식 저장소 코드에서 뜬다.

두 부분으로 나뉜다 (실행 환경이 다르다):

  ref  — AST(audioset) state_dict/forward + MetaCL + Projector 합성 손실.
         참조 환경(torch 1.13 + timm 0.4.5)에서 실행:
           .venv-ref-timm045/bin/python scripts/reproduction/dump_sgscl_method_goldens.py \
               ref --repo <sgscl 클론> --out tests/golden/sgscl_method

  pipeline — 배치 단위 SpecAugment 두 뷰(누적 마스킹) 골든. 원본 util/augmentation.py 는
         nlpaug 를 임포트하므로 스텁을 끼우고 **우리 venv** 에서 실행한다
         (RNG 소모는 numpy/python/torch 인덱싱뿐이라 버전 비의존; patch-mix 파이프라인
         골든과 같은 방식):
           .venv/bin/python scripts/reproduction/dump_sgscl_method_goldens.py \
               pipeline --repo <sgscl 클론> --out tests/golden/sgscl_method

주의:
- MetaCL(method/mcl.py)은 .cuda() 를 하드코딩하므로 CPU 에서 identity 로 패치한다.
- AST 는 ./pretrained_models/audioset_10_10_0.4593.pth (cwd 기준)를 찾는다 —
  아카이브(pretrained_models/upstream/audioset_0.4593.pth)를 심볼릭 링크로 건다.
- Projector 는 det_fill(linspace) 로 파라미터를 덮고 eval 모드(BN 러닝 통계 기본값)
  에서 뜬다 — 양쪽에서 같은 조건이면 합성(projection→MetaCL) 대조로 충분하다.
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

# CPU 참조 환경 대비: .cuda() → identity, torch.load → CPU 강제
torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})


def det_fill(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def dump_ref(repo: pathlib.Path, out: pathlib.Path) -> None:
    # AST audioset ckpt 링크 (원본이 cwd 기준 경로를 하드코딩)
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

    # ---- 1) SG-SCL AST: 스크립트 설정 그대로 (뒤집힌 h/w quirk 포함) ----
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

    # ---- 2) MetaCL 손실 (meta 있음/없음) ----
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

    # ---- 3) 합성: Projector + target_type='project1_project2block' + MetaCL ----
    proj = Projector(16, 16)
    for _, p in sorted(proj.named_parameters()):
        det_fill(p)
    proj.eval()
    torch.manual_seed(303)
    feat1 = torch.randn(6, 16)
    feat2 = torch.randn(6, 16)
    from copy import deepcopy
    p1 = proj(feat1)
    p2 = deepcopy(proj(feat2).detach())  # 원본 main.py:409-411
    comp_loss = mcl(p1, p2, meta)
    torch.save({"feat1": feat1, "feat2": feat2, "meta": meta,
                "p1": p1.detach(), "p2": p2, "loss": comp_loss.detach()},
               out / "sgscl_composition_golden.pt")
    print(f"composition loss {float(comp_loss):.6f}")


def dump_pipeline(repo: pathlib.Path, out: pathlib.Path) -> None:
    # 원본 util/augmentation.py 의 무거운/불필요 임포트 스텁 (마스킹 경로엔 안 쓰임)
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
    x = torch.randn(2, 1, 798, 128)  # [B, 1, time, freq] — 원본 train 배치
    torch.save(x.clone(), out / "twoview_input.pt")

    # 원본 train(): 같은 배치 텐서에 transforms 2회 — in-place 라 마스크가 누적된다
    images = x.clone()
    v1 = transforms(images).clone()
    v2 = transforms(images).clone()
    torch.save({"v1": v1, "v2": v2}, out / "twoview_golden.pt")
    print(f"twoview v1==v2 인 원소 비율 {float((v1 == v2).float().mean()):.4f} "
          f"(마스크 누적이면 1.0 미만이되 v1 마스크 위치는 대부분 유지)")


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
    print("완료:", out)


if __name__ == "__main__":
    main()
