"""
Audio-MAE 백본(DDE-MAE 의 P0 골든) — 공식 facebookresearch/AudioMAE 코드를 별도 venv(timm 0.4.5·CPU) 에서 실행해 저장.

  .venv-ref-timm045/bin/python scripts/reproduction/dump_ddemae_goldens.py \
      --repo outputs/reference/audiomae --ckpt pretrained_models/upstream/audiomae/pretrained.pth --out tests/golden/ddemae

공식 코드는 timm 0.3.2 + timm_patch/swin_transformer.py(디코더 decoder_mode=1 의 SwinTransformerBlock) 를 요구한다 —
여기서는 timm 패키지를 건드리지 않고 timm_patch 파일을 `timm.models.swin_transformer` 이름으로 주입해 실행한다
(util.misc 의 torch._six 는 스텁). numpy≥1.24 의 np.float 제거는 shim.

골든:
- input.pt : x (2,1,1024,128) 고정 난수(정규화 fbank 스케일), noise (2,512) 고정 난수(마스킹), head 결정 초기화 규칙
- mae.pt   : models_mae.mae_vit_base_patch16(audio_exp, img_size (1024,128), decoder_mode 1) + pretrained.pth strict 로드 →
             forward_encoder(x, 0.8, 고정 noise) latent/mask/ids_restore, forward_decoder pred, forward_loss (norm_pix False/True),
             mask_ratio 0(항등 noise) latent, ckpt 키·모양
- vit.pt   : models_vit.vit_base_patch16(global_pool=True, img_size (1024,128), in_chans 1, num_classes 4) 에 ckpt 인코더 로드 →
             forward_features (fc_norm 전역평균) + det_fill head 로짓 + CE
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import torch

if not hasattr(np, "float"):
    np.float = float  # util/pos_embed.py (numpy<1.24 API)


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def _inject_timm_patch(repo: pathlib.Path):
    import timm.models  # noqa: F401
    path = repo / "timm_patch" / "swin_transformer.py"
    spec = importlib.util.spec_from_file_location("timm.models.swin_transformer", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "timm.models"
    sys.modules["timm.models.swin_transformer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_util_misc():
    import util  # noqa: F401  (repo 의 util 패키지)
    stub = types.ModuleType("util.misc")
    stub.concat_all_gather = lambda t: t
    sys.modules["util.misc"] = stub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="outputs/reference/audiomae")
    ap.add_argument("--ckpt", default="pretrained_models/upstream/audiomae/pretrained.pth")
    ap.add_argument("--out", default="tests/golden/ddemae")
    args = ap.parse_args()
    torch.set_num_threads(8)
    repo = pathlib.Path(args.repo).resolve()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    _inject_timm_patch(repo)
    _stub_util_misc()
    import models_mae
    import models_vit

    torch.manual_seed(0)
    x = torch.randn(2, 1, 1024, 128) * 0.5
    torch.manual_seed(1)
    noise = torch.rand(2, 512)
    torch.save({"x": x, "noise": noise, "seed_x": 0, "seed_noise": 1}, out / "input.pt")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]

    # ------------------------------------------------------------------ MAE (인코더+디코더)
    mae = models_mae.mae_vit_base_patch16(audio_exp=True, img_size=(1024, 128), in_chans=1, decoder_mode=1,
                                          norm_pix_loss=False, use_custom_patch=False, mask_2d=False)
    missing, unexpected = mae.load_state_dict(sd, strict=True), None
    mae.eval()

    fixed = {"noise": noise}

    def random_masking_fixed(self, x_, mask_ratio):
        N, L, D = x_.shape
        len_keep = int(L * (1 - mask_ratio))
        nz = fixed["noise"].to(x_.device)
        ids_shuffle = torch.argsort(nz, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x_, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x_.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    mae.random_masking = types.MethodType(random_masking_fixed, mae)
    with torch.no_grad():
        latent, mask, ids_restore, _ = mae.forward_encoder(x, 0.8, mask_2d=False)
        pred, _, _ = mae.forward_decoder(latent, ids_restore)
        loss = mae.forward_loss(x, pred, mask, norm_pix_loss=False)
        loss_np = mae.forward_loss(x, pred, mask, norm_pix_loss=True)
        fixed["noise"] = torch.arange(512, dtype=torch.float32).unsqueeze(0).repeat(2, 1)  # 항등 순서 → 마스크 없음
        latent_full, mask0, ids0, _ = mae.forward_encoder(x, 0.0, mask_2d=False)
        target = mae.patchify(x)
    assert mask0.sum() == 0 and torch.equal(ids0[0], torch.arange(512))
    torch.save({
        "latent": latent, "mask": mask, "ids_restore": ids_restore, "pred": pred, "loss": loss, "loss_norm_pix": loss_np,
        "latent_full": latent_full, "target_patches": target,
        "pos_embed": sd["pos_embed"].clone(), "decoder_pos_embed": sd["decoder_pos_embed"].clone(),
        "ckpt_shapes": {k: list(v.shape) for k, v in sd.items()}, "decoder_mode": 1, "ckpt": args.ckpt,
    }, out / "mae.pt")
    print("mae latent", tuple(latent.shape), "pred", tuple(pred.shape), "loss", float(loss), float(loss_np))

    # ------------------------------------------------------------------ ViT 파인튜닝 경로 (global_pool + fc_norm)
    vit = models_vit.vit_base_patch16(num_classes=4, drop_path_rate=0.0, global_pool=True, mask_2d=False,
                                      use_custom_patch=False, img_size=(1024, 128), in_chans=1)
    # models_vit.VisionTransformer 는 self.norm 을 무조건 삭제(global_pool 은 fc_norm 사용) → norm.* 도 제외
    enc_sd = {k: v for k, v in sd.items() if not (k.startswith("decoder") or k == "mask_token" or k.startswith("norm."))}
    res = vit.load_state_dict(enc_sd, strict=False)
    print("vit load missing", res.missing_keys, "unexpected", res.unexpected_keys)
    assert all(k.startswith("fc_norm") or k.startswith("head") for k in res.missing_keys), res.missing_keys
    assert not res.unexpected_keys
    for _, p in sorted(vit.head.named_parameters()):
        det_fill(p)
    vit.eval()
    with torch.no_grad():
        feat = vit.forward_features(x)
        logits = vit.head(feat)
    y = torch.tensor([0, 2])
    ce = torch.nn.functional.cross_entropy(logits, y)
    torch.save({"feat_gap": feat, "logits": logits, "labels": y, "loss": ce,
                "head_rule": "det_fill(linspace -0.5..0.5) sorted(named_parameters)",
                "vit_keys": sorted(vit.state_dict().keys())}, out / "vit.pt")
    print("vit feat", tuple(feat.shape), "logits", logits.tolist(), "ce", float(ce))
    json.dump({"repo": str(repo), "ckpt": args.ckpt, "timm": __import__("timm").__version__,
               "torch": torch.__version__}, open(out / "meta.json", "w"), indent=1)
    print("완료:", out)


if __name__ == "__main__":
    main()
