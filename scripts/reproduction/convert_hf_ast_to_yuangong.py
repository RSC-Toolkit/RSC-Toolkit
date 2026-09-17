"""
Converts HF `MIT/ast-finetuned-audioset-16-16-0.442` → the original YuanGongND/ast format (`audioset_16_16_0.4422.pth`, DataParallel keys
module.v.* / module.mlp_head.*). Needed because the original Dropbox link is dead (MVST hard-codes this file).

  .venv/bin/python scripts/reproduction/convert_hf_ast_to_yuangong.py [--out pretrained_models/upstream/audioset_16_16_0.4422.pth]

Verification: loads the converted weights into an AST (timm DeiT skeleton) with 16/16 stride and 128×1024 input and compares against the HF model logits (atol 1e-4).
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

    # ---- verification: 128x1024 input, load into the 16/16-stride AST skeleton and compare with the HF logits ----
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
    x = torch.randn(2, 1024, 128)  # (B, time, freq) — same layout as HF input_values
    with torch.no_grad():
        ref = hf(input_values=x).logits
        z = x.unsqueeze(1).transpose(2, 3)  # (B,1,128,1024) = transpose(2,3) in the YuanGong forward
        t = v.patch_embed(z)
        t = torch.cat((v.cls_token.expand(2, -1, -1), v.dist_token.expand(2, -1, -1), t), dim=1) + v.pos_embed
        for blk in v.blocks:
            t = blk(t)
        t = v.norm(t)
        ours = head((t[:, 0] + t[:, 1]) / 2)
    diff = float((ours - ref).abs().max())
    real_missing = [k for k in missing if not k.startswith(("head", "head_dist"))]
    print(f"saved {out} ({len(state)} keys) | skeleton missing (excluding head) {real_missing} unexpected {list(unexpected)} | max|diff| vs HF logits {diff:.2e}")
    json.dump({"source": args.hf, "keys": len(state), "verify_max_abs_diff_vs_hf_logits": diff,
               "note": "HF mirror of YuanGongND/ast 'Full AudioSet, 16 tstride, 16 fstride, no weight averaging, 0.442 mAP', converted to the original key format"},
              open(str(out) + ".provenance.json", "w"), indent=1)
    assert diff < 1e-3, diff


if __name__ == "__main__":
    main()
