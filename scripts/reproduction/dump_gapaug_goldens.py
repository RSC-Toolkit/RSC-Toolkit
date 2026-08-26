"""
GaP-Aug 백본 골든 — 공식 PANNs 코드(qiuqiangkong/audioset_tagging_cnn) 의 Cnn14 를 공식 16 kHz 설정(512/160/64 mel/50–8000)
으로 실행해 특징·임베딩·로짓을 저장한다. GaP-Aug 자체는 코드가 없어 골든이 없다(논문 명세 재구현).

  PYTHONPATH=<scratch pylib(torchlibrosa)> .venv/bin/python scripts/reproduction/dump_gapaug_goldens.py \
      --panns outputs/reference/panns --ckpt pretrained_models/upstream/Cnn14_16k_mAP=0.438.pth --out tests/golden/gapaug

골든 backbone.pt:
  wav        : (3, 160000) — ICBHI 공식 train 사이클 2개(10 s repeat 패딩, 16 kHz) + 고정 seed 잡음 1개
  logmel     : (3, 1, T, 64) — spectrogram_extractor→logmel_extractor 출력
  embedding  : (3, 2048) — eval 모드 fc1 ReLU 출력 (output_dict['embedding'], eval 이라 dropout 없음)
  logits     : (3, 527) — fc_audioset 전 활성(sigmoid 전). fc_audioset.{weight,bias}(PANNs 파라미터 ≈4.3 MB) 는 재배포 회피를
               위해 저장하지 않는다 — 테스트가 ckpt 에서 읽어 대조한다.
  state_keys : 공식 state_dict 키 목록, n_params_total / n_params_wo_audioset
meta.json / provenance.json: PANNs 저장소 커밋 해시·torchlibrosa/torch 버전·ckpt 경로 (opera 골든 선례).
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
    import torchlibrosa  # noqa: E402  (scratch pylib 전용 — 툴킷 런타임 의존성 아님)

    panns_dir = pathlib.Path(args.panns).resolve()
    try:
        panns_commit = subprocess.check_output(["git", "-C", str(panns_dir), "rev-parse", "HEAD"], text=True).strip()
        panns_dirty = bool(subprocess.check_output(["git", "-C", str(panns_dir), "status", "--porcelain"], text=True).strip())
    except Exception:  # git 정보 없으면 기록만 생략
        panns_commit, panns_dirty = None, None

    torch.manual_seed(0)
    # --- 파형: 툴킷 IndexDataset(공식 split train, 16 kHz, 10 s repeat) 의 앞 2개 + 잡음 1개 ---
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
                "note": "fc_audioset.{weight,bias} 는 재배포 회피를 위해 제외 — 테스트가 ckpt 에서 읽는다",
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
