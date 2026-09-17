"""
Official-code parity goldens for OPERA-CT (evelyn0414/OPERA, HTSAT-Swin) + the NTT EMBC'25 frozen evaluation path (eval-audio-repr app/icbhi_sprs) —
runs the official code in our venv (torchlibrosa and efficientnet_pytorch are on unused paths and stubbed; easydict and nnAudio come from a scratch pylib).

  PYTHONPATH=<pylib> .venv/bin/python scripts/reproduction/dump_opera_goldens.py \
      --opera outputs/reference/opera --evar outputs/reference/m2d_evar \
      --ckpt pretrained_models/upstream/opera/encoder-operaCT.ckpt --out tests/golden/opera

Goldens:
- features.pt : batch of sclmeta golden waveforms (ICBHI 6 + SPRS 6, 16 kHz 8 s circular padding) → evar ar_opera.get_entire_signal_librosa
                (batch-level librosa trim → mel64 (1024/512, 50–8000) → dB (ref=batch max, top_db 80) → batch min-max) (B,64,251);
                librosa trim indices of a synthetic batch with silent edges; fixed-seed evar SpecAugment (20,40; mask shared across the batch)
- htsat.pt    : missing/unexpected keys when loading the ckpt into the official Cola(encoder="htsat") (strict=False), htsat state_dict keys/shapes,
                eval embeddings extract_feature(dim=768) (B,768), fixed-seed train-mode embeddings (bn0 batch statistics + DropPath) and the bn0 running statistics afterwards
- head.pt     : state_dict and eval logits of the fixed-seed TfmEncHead(768→n_cls, depth 4, heads 1, mlp_ratio 1) / MLP(768,(128,),n_cls)
"""
import argparse
import json
import pathlib
import sys
import types

import numpy as np
import torch
import torch.nn as nn


def _install_stubs():
    class _Id(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, x):
            return x

    tl = types.ModuleType("torchlibrosa")
    st = types.ModuleType("torchlibrosa.stft")
    au = types.ModuleType("torchlibrosa.augmentation")
    st.Spectrogram = _Id
    st.LogmelFilterBank = _Id
    au.SpecAugmentation = _Id
    sys.modules.update({"torchlibrosa": tl, "torchlibrosa.stft": st, "torchlibrosa.augmentation": au})
    ef = types.ModuleType("efficientnet_pytorch")
    ef.EfficientNet = type("EfficientNet", (), {"from_name": staticmethod(lambda *a, **k: nn.Identity())})
    sys.modules["efficientnet_pytorch"] = ef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opera", required=True)
    ap.add_argument("--evar", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="tests/golden/opera")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(args.opera).resolve()))
    sys.path.insert(0, str(pathlib.Path(args.evar).resolve()))
    _install_stubs()
    import librosa
    from src.model.models_cola import Cola
    import evar.ar_opera as ao
    # evar/finetune.py pulls in evar.data (samplers etc.) and is heavy to import, so only the SpecAugment class source is exec'd as-is
    import torchaudio
    src = (pathlib.Path(args.evar) / "finetune.py").read_text()
    block = src[src.index("class SpecAugment:"):src.index("class AudioFineuneAug")]
    ns = {"torchaudio": torchaudio}
    exec(block, ns)
    SpecAugment = ns["SpecAugment"]

    torch.use_deterministic_algorithms(True, warn_only=True)

    # ---------------- inputs ----------------
    icbhi = torch.stack([w["wav"] for w in torch.load("tests/golden/sclmeta/audio.pt")])          # (6, 128000)
    sprs = torch.stack([w["wav"] for w in torch.load("tests/golden/sclmeta_sprs/audio.pt")])      # (6, 128000)
    assert icbhi.shape == sprs.shape == (6, 128000)
    # synthetic batch with silent edges: zero the first 0.5 s / last 0.3 s (creates a case where the original trim actually cuts)
    synth = icbhi.clone()
    synth[:, : 8000] = 0.0
    synth[:, -4800:] = 0.0
    FRAME_LEN, HOP = 1600, 800
    trims = {}
    for name, x in (("icbhi", icbhi), ("sprs", sprs), ("synth", synth)):
        _, idx = librosa.effects.trim(x.numpy(), frame_length=FRAME_LEN, hop_length=HOP)
        trims[name] = [int(idx[0]), int(idx[1])]
    print("trim indices", trims)
    assert trims["icbhi"] == [0, 128000] and trims["sprs"] == [0, 128000], "if the golden waveforms get trimmed, the original path breaks"

    with torch.no_grad():
        feat_icbhi = ao.get_entire_signal_librosa(icbhi, input_sec=8)  # (6, 64, 251) float32
        feat_sprs = ao.get_entire_signal_librosa(sprs, input_sec=8)
    print("features", tuple(feat_icbhi.shape), feat_icbhi.dtype, float(feat_icbhi.min()), float(feat_icbhi.max()))
    # record the batch-split dependence: features from only the first 3 (batch statistics change)
    with torch.no_grad():
        feat_icbhi_half = ao.get_entire_signal_librosa(icbhi[:3], input_sec=8)

    specaug = SpecAugment(20, 40)
    torch.manual_seed(5)
    aug1 = specaug(feat_icbhi.clone())
    aug2 = specaug(feat_icbhi.clone())
    torch.save(
        {
            "wav_icbhi": icbhi, "wav_sprs": sprs, "wav_synth": synth, "trim_indices": trims,
            "feat_icbhi": feat_icbhi, "feat_sprs": feat_sprs, "feat_icbhi_half": feat_icbhi_half,
            "specaug_seed": 5, "specaug_view1": aug1, "specaug_view2": aug2,
        },
        out / "features.pt",
    )

    # ---------------- backbone ----------------
    model = Cola(encoder="htsat")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    res = model.load_state_dict(ck["state_dict"], strict=False)
    htsat = model.encoder.encoder.htsat
    sd = htsat.state_dict()
    model.eval()
    with torch.no_grad():
        emb_icbhi = model.extract_feature(feat_icbhi.transpose(-2, -1), 768)  # (6, 768)
        emb_sprs = model.extract_feature(feat_sprs.transpose(-2, -1), 768)
    bn_before = {k: v.clone() for k, v in htsat.bn0.state_dict().items()}
    model.train()
    torch.manual_seed(11)
    with torch.no_grad():
        emb_train = model.extract_feature(feat_icbhi.transpose(-2, -1), 768)
    bn_after = {k: v.clone() for k, v in htsat.bn0.state_dict().items()}
    torch.save(
        {
            "missing_keys": res.missing_keys, "unexpected_keys": res.unexpected_keys,
            "ckpt_keys": sorted(ck["state_dict"].keys()),
            "htsat_keys": {k: list(v.shape) for k, v in sd.items()},
            "htsat_n_params": sum(p.numel() for p in htsat.parameters()),
            "emb_icbhi": emb_icbhi, "emb_sprs": emb_sprs,
            "train_seed": 11, "emb_train": emb_train, "bn0_before": bn_before, "bn0_after": bn_after,
        },
        out / "htsat.pt",
    )
    print("emb", tuple(emb_icbhi.shape), "missing", len(res.missing_keys), "unexpected", res.unexpected_keys)

    # ---------------- heads (solve.py) ----------------
    from timm.models.vision_transformer import Block

    class TfmEncHead(nn.Module):  # solve.py as-is
        def __init__(self, embed_dim=768, output_dim=3, depth=2, heads=1, mlp_ratio=1):
            super().__init__()
            self.sem_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.sem_blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for i in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            self.apply(self._init_weights)
            self.linear = nn.Linear(embed_dim, output_dim)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def forward(self, x):
            sem_tokens = self.sem_token.expand(x.shape[0], -1, -1)
            x = torch.cat((sem_tokens, x), dim=1)
            for blk in self.sem_blocks:
                x = blk(x)
            x = x[:, 0, :]
            x = self.norm(x)
            return self.linear(x)

    class MLP(nn.Module):  # solve.py as-is
        def __init__(self, input_size, hidden_sizes, output_size, hidden_dropout=0.5, mean=0.0, std=0.01, bias=0.):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(input_size, affine=False)
            sizes = [input_size] + list(hidden_sizes) + [output_size]
            fcs = []
            for l, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
                if l > 0:
                    fcs.append(nn.Dropout(hidden_dropout))
                linear = nn.Linear(in_size, out_size)
                nn.init.normal_(linear.weight, mean=mean, std=std)
                nn.init.constant_(linear.bias, bias)
                fcs.append(linear)
                fcs.append(nn.ReLU())
            self.mlp = nn.Sequential(*fcs[:-1])

        def forward(self, x):
            x = torch.mean(x, dim=-2)
            x = self.norm(x.unsqueeze(-1)).squeeze(-1)
            return self.mlp(x)

    tokens = emb_icbhi.unsqueeze(1)  # encode_frames [B,D,1] → transpose → [B,1,D]
    torch.manual_seed(3)
    tfm = TfmEncHead(embed_dim=768, output_dim=4, depth=4, heads=1, mlp_ratio=1).eval()
    torch.manual_seed(4)
    mlp = MLP(768, (128,), 4).eval()
    with torch.no_grad():
        logits_tfm = tfm(tokens)
        logits_mlp = mlp(tokens)
    torch.save(
        {
            # instead of the full state_dict (57 MB), a per-key summary of shape, sum and mean absolute value — compared against the toolkit head built with the same seed
            "tfm_seed": 3, "tfm_state_meta": {k: [list(v.shape), float(v.double().sum()), float(v.double().abs().mean())] for k, v in tfm.state_dict().items()},
            "logits_tfm": logits_tfm,
            "mlp_seed": 4, "mlp_state_meta": {k: [list(v.shape), float(v.double().sum()), float(v.double().abs().mean())] for k, v in mlp.state_dict().items()},
            "logits_mlp": logits_mlp,
        },
        out / "head.pt",
    )
    json.dump(
        {"opera": str(pathlib.Path(args.opera).resolve()), "evar": str(pathlib.Path(args.evar).resolve()), "ckpt": args.ckpt,
         "torch": torch.__version__, "librosa": librosa.__version__, "n_htsat_keys": len(sd)},
        open(out / "provenance.json", "w"), indent=2,
    )
    print("done", out)


if __name__ == "__main__":
    main()
