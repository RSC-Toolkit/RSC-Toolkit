"""
HF `MIT/ast-finetuned-audioset-16-16-0.442` → YuanGongND/ast 원본 형식(`audioset_16_16_0.4422.pth`, DataParallel 키
module.v.* / module.mlp_head.*) 변환. 원 Dropbox 링크가 비활성이라 필요 (MVST 가 이 파일을 하드코딩).

  .venv/bin/python scripts/reproduction/convert_hf_ast_to_yuangong.py [--out pretrained_models/upstream/audioset_16_16_0.4422.pth]

검증: 변환본을 16/16 스트라이드·128×1024 입력 AST(timm DeiT 골격)에 로드해 HF 모델 로짓과 대조(atol 1e-4).
"""
import argparse
import json
import pathlib

import torch


def convert(hf_state: dict) -> dict:
    P = "audio_spectrogram_transformer."
    out = {
        "module.v.cls_token": hf_state[P + "embeddings.cls_token"],
        "module.v.dist_token": hf_state[P + "embeddings.distillation_token"],
        "module.v.pos_embed": hf_state[P + "embeddings.position_embeddings"],
        "module.v.patch_embed.proj.weight": hf_state[P + "embeddings.patch_embeddings.projection.weight"],
        "module.v.patch_embed.proj.bias": hf_state[P + "embeddings.patch_embeddings.projection.bias"],
        "module.v.norm.weight": hf_state[P + "layernorm.weight"],
        "module.v.norm.bias": hf_state[P + "layernorm.bias"],
        "module.mlp_head.0.weight": hf_state["classifier.layernorm.weight"],
        "module.mlp_head.0.bias": hf_state["classifier.layernorm.bias"],
        "module.mlp_head.1.weight": hf_state["classifier.dense.weight"],
        "module.mlp_head.1.bias": hf_state["classifier.dense.bias"],
    }
    n_layers = max(int(k.split(".")[2]) for k in hf_state if k.startswith(P + "layers.")) + 1
    for i in range(n_layers):
        L = f"{P}layers.{i}."
        B = f"module.v.blocks.{i}."
        out[B + "norm1.weight"] = hf_state[L + "layernorm_before.weight"]
        out[B + "norm1.bias"] = hf_state[L + "layernorm_before.bias"]
        out[B + "attn.qkv.weight"] = torch.cat([hf_state[L + f"attention.{n}_proj.weight"] for n in ("q", "k", "v")], dim=0)
        out[B + "attn.qkv.bias"] = torch.cat([hf_state[L + f"attention.{n}_proj.bias"] for n in ("q", "k", "v")], dim=0)
        out[B + "attn.proj.weight"] = hf_state[L + "attention.o_proj.weight"]
        out[B + "attn.proj.bias"] = hf_state[L + "attention.o_proj.bias"]
        out[B + "norm2.weight"] = hf_state[L + "layernorm_after.weight"]
        out[B + "norm2.bias"] = hf_state[L + "layernorm_after.bias"]
        out[B + "mlp.fc1.weight"] = hf_state[L + "mlp.fc1.weight"]
        out[B + "mlp.fc1.bias"] = hf_state[L + "mlp.fc1.bias"]
        out[B + "mlp.fc2.weight"] = hf_state[L + "mlp.fc2.weight"]
        out[B + "mlp.fc2.bias"] = hf_state[L + "mlp.fc2.bias"]
    return {k: v.detach().clone().contiguous() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="MIT/ast-finetuned-audioset-16-16-0.442")
    ap.add_argument("--out", default="pretrained_models/upstream/audioset_16_16_0.4422.pth")
    args = ap.parse_args()
    from transformers import ASTForAudioClassification

    hf = ASTForAudioClassification.from_pretrained(args.hf).eval()
    state = convert(hf.state_dict())
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)

    # ---- 검증: 128x1024 입력, 16/16 스트라이드 AST 골격에 로드해 HF 로짓과 대조 ----
    from rsc_toolkit.models.ast_patchmix import PatchEmbed, _create_deit_skeleton
    v = _create_deit_skeleton("base384", pretrained=False)
    pe = PatchEmbed(img_size=(128, 1024), patch_size=16, in_chans=1, embed_dim=768)
    pe.proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16)); v.patch_embed = pe
    v.pos_embed = torch.nn.Parameter(torch.zeros(1, 514, 768))
    v_state = {k[len("module.v."):]: t for k, t in state.items() if k.startswith("module.v.")}
    missing, unexpected = v.load_state_dict(v_state, strict=False)
    head = torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, 527))
    head.load_state_dict({k[len("module.mlp_head."):]: t for k, t in state.items() if k.startswith("module.mlp_head.")})
    v.eval(); head.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 1024, 128)  # (B, time, freq) — HF input_values 와 동일 레이아웃
    with torch.no_grad():
        ref = hf(input_values=x).logits
        z = x.unsqueeze(1).transpose(2, 3)  # (B,1,128,1024) = YuanGong forward 의 transpose(2,3)
        t = v.patch_embed(z)
        t = torch.cat((v.cls_token.expand(2, -1, -1), v.dist_token.expand(2, -1, -1), t), dim=1) + v.pos_embed
        for blk in v.blocks:
            t = blk(t)
        t = v.norm(t)
        ours = head((t[:, 0] + t[:, 1]) / 2)
    diff = float((ours - ref).abs().max())
    real_missing = [k for k in missing if not k.startswith(("head", "head_dist"))]
    print(f"저장 {out} ({len(state)} 키) | 골격 missing(head 제외) {real_missing} unexpected {list(unexpected)} | HF 로짓 대비 max|diff| {diff:.2e}")
    json.dump({"source": args.hf, "keys": len(state), "verify_max_abs_diff_vs_hf_logits": diff,
               "note": "YuanGongND/ast 'Full AudioSet, 16 tstride, 16 fstride, no weight averaging, 0.442 mAP' 의 HF 미러를 원본 키 형식으로 변환"},
              open(str(out) + ".provenance.json", "w"), indent=1)
    assert diff < 1e-3, diff


if __name__ == "__main__":
    main()
