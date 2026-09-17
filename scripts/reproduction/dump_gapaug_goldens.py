"""
GaP-Aug backbone goldens — runs Cnn14 from the official PANNs code (qiuqiangkong/audioset_tagging_cnn) with the official 16 kHz settings (512/160/64 mel/50–8000)
and saves features, embeddings and logits. GaP-Aug itself has no code and therefore no goldens (paper-spec re-implementation).

  PYTHONPATH=<scratch pylib(torchlibrosa)> .venv/bin/python scripts/reproduction/dump_gapaug_goldens.py \
      --panns outputs/reference/panns --ckpt pretrained_models/upstream/Cnn14_16k_mAP=0.438.pth --out tests/golden/gapaug

Goldens in backbone.pt:
  wav        : (3, 160000) — 2 ICBHI official train cycles (10 s repeat padding, 16 kHz) + 1 fixed-seed noise clip
  logmel     : (3, 1, T, 64) — spectrogram_extractor→logmel_extractor output
  embedding  : (3, 2048) — eval-mode fc1 ReLU output (output_dict['embedding']; no dropout in eval)
  logits     : (3, 527) — fc_audioset output before the sigmoid. fc_audioset.{weight,bias} (PANNs parameters, ≈4.3 MB) are not saved
               to avoid redistribution — the test reads them from the ckpt for the comparison.
  state_keys : list of official state_dict keys, n_params_total / n_params_wo_audioset
meta.json / provenance.json: PANNs repository commit hash, torchlibrosa/torch versions, ckpt path (same convention as the OPERA goldens).
"""
import argparse
import json
import pathlib
import subprocess
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panns", default="outputs/reference/panns")
    ap.add_argument("--ckpt", default="pretrained_models/upstream/Cnn14_16k_mAP=0.438.pth")
    ap.add_argument("--out", default="tests/golden/gapaug")
    ap.add_argument("--data", default=".datasets/icbhi")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(args.panns).resolve() / "pytorch"))
    sys.path.insert(0, str(pathlib.Path(args.panns).resolve() / "utils"))
    from models import Cnn14  # noqa: E402
    import torchlibrosa  # noqa: E402  (scratch pylib only — not a toolkit runtime dependency)

    panns_dir = pathlib.Path(args.panns).resolve()
    try:
        panns_commit = subprocess.check_output(["git", "-C", str(panns_dir), "rev-parse", "HEAD"], text=True).strip()
        panns_dirty = bool(subprocess.check_output(["git", "-C", str(panns_dir), "status", "--porcelain"], text=True).strip())
    except Exception:  # if git info is unavailable, just skip recording it
        panns_commit, panns_dirty = None, None

    torch.manual_seed(0)
    # --- waveforms: first 2 items of the toolkit IndexDataset (official split train, 16 kHz, 10 s repeat) + 1 noise clip ---
    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    data = pathlib.Path(args.data)
    pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=10.0, pad_type="repeat"))
    ds = IndexDataset(index_csv=data / "index/index_cycle.csv", data_root=data, sampling_rate=16000, sample_level="cycle",
                      split="train", split_file="rsc_toolkit/datasets/icbhi/data/official_split.txt", audio_preprocessor=pre)
    wavs, ids, labels = [], [], []
    for i in (0, 1):
        it = ds[i]
        wavs.append(it["audio"].reshape(-1)); ids.append(it["sample_id"]); labels.append(int(it["label"]))
    wavs.append(torch.randn(160000) * 0.05); ids.append("noise_seed0"); labels.append(-1)
    wav = torch.stack(wavs).float()

    model = Cnn14(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        x = model.spectrogram_extractor(wav)
        logmel = model.logmel_extractor(x)
        od = model(wav)
        emb = od["embedding"]
        logits = model.fc_audioset(emb)
    sd = model.state_dict()
    n_total = sum(v.numel() for v in sd.values())
    n_wo = sum(v.numel() for k, v in sd.items() if not k.startswith(("fc_audioset.", "spectrogram_extractor.", "logmel_extractor.")))
    torch.save({"wav": wav, "sample_ids": ids, "labels": labels, "logmel": logmel, "embedding": emb, "logits": logits,
                "note": "fc_audioset.{weight,bias} excluded to avoid redistribution — the test reads them from the ckpt",
                "state_keys": list(sd.keys()), "state_shapes": {k: list(v.shape) for k, v in sd.items()},
                "n_params_total": n_total, "n_params_wo_audioset_and_frontend": n_wo,
                "config": dict(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000)},
               out / "backbone.pt")
    prov = {"panns_repo": "https://github.com/qiuqiangkong/audioset_tagging_cnn", "panns_dir": str(panns_dir),
            "panns_commit": panns_commit, "panns_dirty": panns_dirty, "panns_license": "MIT",
            "ckpt": args.ckpt, "ckpt_zenodo": "https://zenodo.org/record/3987831 (Cnn14_16k_mAP=0.438.pth)",
            "torch": torch.__version__, "torchlibrosa": getattr(torchlibrosa, "__version__", None),
            "config": dict(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000)}
    json.dump({"ckpt": args.ckpt, "sample_ids": ids, "logmel_shape": list(logmel.shape), "n_params_total": n_total,
               "n_params_wo_audioset_and_frontend": n_wo, "torch": torch.__version__, "torchlibrosa": prov["torchlibrosa"],
               "panns_commit": panns_commit, "fc_audioset_in_golden": False},
              open(out / "meta.json", "w"), indent=1)
    json.dump(prov, open(out / "provenance.json", "w"), indent=2)
    print("saved", out / "backbone.pt", logmel.shape, emb.shape, n_total, n_wo)


if __name__ == "__main__":
    main()
