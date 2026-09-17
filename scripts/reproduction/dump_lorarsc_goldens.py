"""
LoRA-RSC (ICASSP 2025, deegy666/LoRA-RSC d1f0653) official-code parity goldens — runs the official code in a reference venv with the README-pinned environment (transformers 4.28.1).

  uv venv .venv-ref-lorarsc --python 3.10
  uv pip install --python .venv-ref-lorarsc/bin/python 'torch==2.4.1+cpu' 'torchaudio==2.4.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv-ref-lorarsc/bin/python 'transformers==4.28.1' 'huggingface_hub==0.25.2' 'numpy==1.26.4' safetensors pyyaml
  CUDA_VISIBLE_DEVICES= .venv-ref-lorarsc/bin/python scripts/reproduction/dump_lorarsc_goldens.py \
      --repo outputs/reference/lorarsc --hf pretrained_models/upstream/hf/ast-finetuned-audioset-10-10-0.4593 --out tests/golden/lorarsc

Golden contents:
  features.pt : waveforms from applying the original dataset/icbhi.py get_sample/get_dataset chain (segment load→16k resample→Fade(sr/16)→8 s circular padding+fade_out)
                as-is to 4 cycles of toy_dataset, the np.random.randn(len(audio)) noise values (scalar DC offset), and the
                4.28.1 ASTFeatureExtractor(max_length=500) outputs (with/without noise) (N,500,128). Also records the result of the original 2D (1,N) array call (whether it raises).
  model.pt    : official AST_LoRA (max_length 500, CLS, rank=96 (=reduction), alpha 8) — LoRA weight statistics right after from_pretrained
                (checks the fast-init re-initialization quirk: not zero), pos_embed zero check, parameter counts (total/trainable), state_dict keys/shapes,
                eval forward logits and CE loss after making LoRA and head deterministic with det_fill. full-FT (src/AST.py) is recorded the same way.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

TOY = [("107_2b3_Al_mc_AKGC417L", [(0.887, 2.601, 1), (2.601, 5.327, 1)]),
       ("107_2b3_Ar_mc_AKGC417L", None)]  # second file: first 2 cycles from its .txt


def det_fill(t, scale=1.0):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape) * scale)


def read_cycles(txt, n=2):
    rows = [l.split() for l in open(txt) if l.strip()][:n]
    return [(float(a), float(b), (0 if int(w) == 0 and int(c) == 0 else 1 if int(w) == 0 else 2 if int(c) == 0 else 3))
            for a, b, c, w in rows]


def original_waveform(path, onset, offset, sr_target=16000, duration=8):
    """dataset/icbhi.py get_sample + get_dataset(pad_type='circular') as-is (librosa.get_samplerate → torchaudio.info)."""
    sr = torchaudio.info(str(path)).sample_rate
    audio, _ = torchaudio.load(str(path), int(onset * sr), (int(offset * sr) - int(onset * sr)))
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sr_target:
        audio = T.Resample(sr, sr_target)(audio)
    fade_samples = int(sr_target / 16)
    audio = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear")(audio)
    target = int(duration * sr_target)
    if audio.shape[-1] > target:
        audio = audio[..., :target]
    else:
        ratio = math.ceil(target / audio.shape[-1])
        audio = audio.repeat(1, ratio)[..., :target]
        audio = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape="linear")(audio)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--data", default="sample_data/toy_dataset/files")
    ap.add_argument("--out", default="tests/golden/lorarsc")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(args.repo).resolve()))
    import transformers
    from transformers import ASTFeatureExtractor
    from src.AST_LoRA_ICBHI import AST_LoRA
    from src.AST import AST

    assert transformers.__version__ == "4.28.1", transformers.__version__

    # ---------------- (a) feature extraction ----------------
    processor = ASTFeatureExtractor.from_pretrained(args.hf, max_length=500)
    assert processor.max_length == 500
    waveforms, labels, feats_noise, feats_clean, noises, meta = [], [], [], [], [], []
    np.random.seed(10)
    orig_2d_call = None
    for fname, cycles in TOY:
        wav = pathlib.Path(args.data) / f"{fname}.wav"
        cycles = cycles or read_cycles(pathlib.Path(args.data) / f"{fname}.txt")
        for onset, offset, label in cycles:
            audio = original_waveform(wav, onset, offset)
            audio_numpy = audio.numpy().astype(np.float64)  # (1, 128000)
            noise = np.random.randn(len(audio_numpy))      # len == 1 → a single scalar
            augmented = audio_numpy + 0.3 * noise
            if orig_2d_call is None:  # record whether the original 2D (1,N) call actually works
                try:
                    r = processor(augmented, sampling_rate=16000, return_tensors="pt")["input_values"]
                    orig_2d_call = {"ok": True, "shape": list(r.shape)}
                except Exception as e:  # noqa: BLE001
                    orig_2d_call = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            fn = processor(augmented[0], sampling_rate=16000, return_tensors="pt")["input_values"]  # (1,500,128)
            fc = processor(audio_numpy[0], sampling_rate=16000, return_tensors="pt")["input_values"]
            waveforms.append(audio); labels.append(label); noises.append(float(noise[0]))
            feats_noise.append(fn); feats_clean.append(fc)
            meta.append({"file": fname, "onset": onset, "offset": offset, "label": label, "noise_shape": list(noise.shape)})
    waveforms = torch.cat(waveforms, 0)              # (N, 128000)
    feats_noise = torch.cat(feats_noise, 0)          # (N, 500, 128)
    feats_clean = torch.cat(feats_clean, 0)
    torch.save({"waveforms": waveforms, "labels": torch.tensor(labels), "noise_scalar": torch.tensor(noises),
                "features_noise": feats_noise, "features_clean": feats_clean, "meta": meta, "orig_2d_call": orig_2d_call},
               out / "features.pt")
    print("2D call:", orig_2d_call, "| feature max|diff| with/without noise:", float((feats_noise - feats_clean).abs().max()))

    # ---------------- (b)(c)(d) model ----------------
    x = feats_noise.unsqueeze(1)  # original dataset: (N,1,500,128); AST_LoRA.forward applies squeeze(1)
    y = torch.tensor(labels)
    torch.manual_seed(10)
    model = AST_LoRA(max_length=500, num_classes=4, final_output="CLS", rank=96, alpha=8, model_ckpt=args.hf)
    sa0 = model.encoder.layer[0].attention.attention
    init_stats = {n: {"std": float(getattr(sa0, n).weight.std()), "absmax": float(getattr(sa0, n).weight.abs().max()),
                      "shape": list(getattr(sa0, n).weight.shape)}
                  for n in ("lora_down_k", "lora_up_k", "lora_down_v", "lora_up_v")}
    pos = model.embeddings.position_embeddings
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    # make deterministic: LoRA 4×12 + head (down/up at a small scale)
    lora_state = {}
    for i, layer in enumerate(model.encoder.layer):
        sa = layer.attention.attention
        for n, s in (("lora_down_k", 0.05), ("lora_up_k", 0.05), ("lora_down_v", 0.05), ("lora_up_v", 0.05)):
            w = getattr(sa, n).weight
            det_fill(w, s)
            lora_state[f"encoder.layer.{i}.attention.attention.{n}.weight"] = w.detach().clone()
    det_fill(model.classification_head.weight, 0.1)
    det_fill(model.classification_head.bias, 0.1)
    head_state = {k: v.detach().clone() for k, v in model.classification_head.state_dict().items()}
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        # must also match in training mode (the original train() override keeps encoder/embeddings in eval) (dropout 0)
        model.train(True)
        logits_train = model(x)
        model.eval()
    sd_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

    torch.manual_seed(10)
    full = AST(max_length=500, num_classes=4, final_output="CLS", model_ckpt=args.hf)
    det_fill(full.classification_head.weight, 0.1)
    det_fill(full.classification_head.bias, 0.1)
    full.eval()
    with torch.no_grad():
        logits_full = full(x)
        loss_full = nn.CrossEntropyLoss()(logits_full, y)
    full_stats = {"n_total": sum(p.numel() for p in full.parameters()),
                  "n_train": sum(p.numel() for p in full.parameters() if p.requires_grad),
                  "pos_absmax": float(full.model.embeddings.position_embeddings.abs().max()),
                  "pos_shape": list(full.model.embeddings.position_embeddings.shape)}

    torch.save({
        "input": x, "labels": y,
        "lora_init_stats_layer0": init_stats, "pos_embed_absmax": float(pos.abs().max()), "pos_embed_shape": list(pos.shape),
        "pos_embed_requires_grad": bool(pos.requires_grad),
        "n_total": n_total, "n_train": n_train, "trainable_names": trainable_names, "state_dict_shapes": sd_shapes,
        "scaling": float(sa0.scaling), "rank_dim": sa0.lora_down_k.weight.shape[0],
        "lora_state": lora_state, "head_state": head_state,
        "logits": logits, "loss": loss, "logits_train_mode": logits_train,
        "full_ft": {"logits": logits_full, "loss": loss_full, "head_state": {k: v.detach().clone() for k, v in full.classification_head.state_dict().items()}, **full_stats},
    }, out / "model.pt")
    json.dump({"repo": str(pathlib.Path(args.repo).resolve()), "hf": args.hf, "torch": torch.__version__,
               "torchaudio": torchaudio.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
              open(out / "provenance.json", "w"), indent=2)
    print("LoRA init(layer0):", init_stats)
    print("pos_embed absmax:", float(pos.abs().max()), "n_total:", n_total, "n_train:", n_train, "scaling:", sa0.scaling)
    print("logits:", logits, "loss:", float(loss), "| train-mode max|diff|:", float((logits - logits_train).abs().max()))
    print("full-FT:", full_stats, "loss:", float(loss_full))


if __name__ == "__main__":
    main()
