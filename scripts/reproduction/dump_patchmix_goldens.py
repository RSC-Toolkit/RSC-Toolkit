"""
Produces official-code parity golden files from the official Patch-Mix code (timm 0.4.5).

Run (the reference environment is required — an old-version environment, not the toolkit venv):
  .venv-ref-timm045/bin/python scripts/reproduction/dump_patchmix_goldens.py \
      --repo <path to the patch-mix repository clone> --ckpt pretrained_models/upstream/audioset_0.4593.pth

Outputs (tests/golden/patchmix/):
  - reference_versions.json  : torch/timm versions of the reference environment
  - state_dict_shapes.json   : {key: shape} of the reference ASTModel(audioset, 798/128)
  - features_input.pt / features_golden.pt : fixed input and reference forward output (B=2, 768)
  - loss_goldens.pt          : loss input tensors + reference loss values

Notes:
- The reference PatchMixConLoss hard-codes .cuda(), so for CPU execution
  Tensor.cuda / Module.cuda are patched to identity (no effect on the math).
- The input tensors are saved as well, so RNG differences across versions do not affect the results.
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

# neutralize the hard-coded .cuda() of the reference code in a CPU-only environment
torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
nn.Module.cuda = lambda self, *a, **k: self  # type: ignore[assignment]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the patch-mix official repository clone")
    ap.add_argument("--ckpt", required=True, help="path to audioset_0.4593.pth")
    ap.add_argument("--out", default="tests/golden/patchmix")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    ckpt = pathlib.Path(args.ckpt).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import subprocess
    import timm

    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    # record the reference repository by URL and commit — the local clone path differs per machine and is not provenance of the goldens
    json.dump(
        {"torch": torch.__version__, "timm": timm.__version__,
         "repo": _git("config", "--get", "remote.origin.url") or repo.name, "commit": _git("rev-parse", "--short", "HEAD")},
        open(out / "reference_versions.json", "w"), indent=1,
    )

    sys.path.insert(0, str(repo))

    # the reference ast.py looks for './pretrained_models/audioset_10_10_0.4593.pth' relative to cwd
    workdir = tempfile.mkdtemp(prefix="patchmix_golden_")
    os.makedirs(os.path.join(workdir, "pretrained_models"), exist_ok=True)
    os.symlink(ckpt, os.path.join(workdir, "pretrained_models", "audioset_10_10_0.4593.pth"))
    os.chdir(workdir)

    from models.ast import ASTModel  # noqa: E402  (reference repository)
    from method.patchmix import PatchMixLoss  # noqa: E402
    from method.patchmix_cl import PatchMixConLoss  # noqa: E402

    # ---- model goldens: same as the AST call in main.py (including the 798/128 swap) ----
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

    # ---- loss goldens (inputs saved too) ----
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
    print("done:", out)


if __name__ == "__main__":
    main()
