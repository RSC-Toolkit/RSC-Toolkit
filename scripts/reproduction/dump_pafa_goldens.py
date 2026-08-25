"""
PAFA(wa976/PAFA) 의 P0 골든 — 공식 코드를 우리 venv 에서 실행해 뜬다 (torch/torchaudio 만 필요).

  .venv/bin/python scripts/reproduction/dump_pafa_goldens.py --repo <PAFA 클론> --out tests/golden/pafa

골든:
- beats_features.pt   : 원본 BEATsTransferLearningModel(iter3+ AS2M ckpt, eval) 의 (2, 80000) 파형 → (2, T', 768)
- pafa_head_loss.pt   : 원본 ProjectionHead(768→768, attention, ln; det_fill) + PAFALoss(λ 50 / 5e-4) 출력
- pipeline_golden.pt  : 원본 슬라이스·cut/pad(5초 repeat+fade) 를 실제 ICBHI 녹음 2개에 적용한 cycle 파형
"""
import argparse
import pathlib
import sys
import types

import torch


def det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="tests/golden/pafa")
    ap.add_argument("--ckpt", default="pretrained_models/upstream/BEATs_iter3_plus_AS2M.pt")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    # 원본 util 의 무관한 임포트 스텁 (torch dynamo 가 sys.modules 를 훑으므로 __spec__ 을 준다)
    import importlib.machinery
    for name in ("tkinter", "nlpaug", "nlpaug.augmenter", "nlpaug.augmenter.audio", "cmapy", "wget",
                 "matplotlib", "matplotlib.pyplot", "seaborn", "sklearn", "sklearn.manifold"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
            sys.modules[name] = mod
    sys.modules["tkinter"].W = "w"

    # ---- 1) BEATs 특징 ----
    from models.beats import BEATsTransferLearningModel  # noqa: E402
    m = BEATsTransferLearningModel(num_target_classes=4, model_path=args.ckpt, spec_transform=None).eval()
    torch.manual_seed(7)
    wav = torch.randn(2, 80000) * 0.1
    with torch.no_grad():
        feats = m(wav)
    torch.save({"wav": wav, "features": feats}, out / "beats_features.pt")
    print("beats features", tuple(feats.shape), float(feats.abs().mean()))

    # ---- 2) ProjectionHead + PAFALoss ----
    from method.pafa import ProjectionHead, PAFALoss  # noqa: E402
    head = ProjectionHead(768, None, 768, attention=True, norm_type="ln", proj_type="end2end")
    for _, p in sorted(head.named_parameters()):
        det_fill(p)
    head.eval()
    torch.manual_seed(8)
    f = torch.randn(6, 5, 768)                      # (B, T', D) 토큰 시퀀스
    pid = torch.tensor([101, 101, 102, 103, 103, 103])
    with torch.no_grad():
        proj = head(f)
    loss = PAFALoss()(proj, pid, lambda_pcsl=50.0, lambda_gpal=0.0005)
    loss_single = PAFALoss()(proj, torch.full_like(pid, 5), lambda_pcsl=50.0, lambda_gpal=0.0005)
    torch.save({"features": f, "patient_ids": pid, "proj": proj, "loss": loss.detach(), "loss_single": loss_single.detach()},
               out / "pafa_head_loss.pt")
    print("pafa loss", float(loss), "single-patient", float(loss_single))

    # ---- 3) 데이터 파이프라인 (원본 slice + cut_pad, 실제 녹음) ----
    import torchaudio  # noqa: E402
    from torchaudio import transforms as T  # noqa: E402
    from util.icbhi_util import _slice_data_torchaudio, cut_pad_sample_torchaudio  # noqa: E402
    import pandas as pd, json  # noqa: E402
    a = types.SimpleNamespace(sample_rate=16000, desired_length=5, pad_types="repeat")
    idx = pd.read_csv(".datasets/icbhi/index/index_cycle.csv")
    recs = ["101_1b1_Al_sc_Meditron", "226_1b1_Pl_sc_LittC2SE"]
    rows = idx[idx["recording_path"].str.contains("|".join(recs))].head(8)
    golden = []
    for _, r in rows.iterrows():
        fpath = pathlib.Path(".datasets/icbhi") / r["recording_path"]
        data, sr = torchaudio.load(str(fpath))
        if sr != a.sample_rate:
            data = T.Resample(sr, a.sample_rate)(data)
        fs = int(a.sample_rate / 16)
        data = T.Fade(fade_in_len=fs, fade_out_len=fs, fade_shape="linear")(data)
        chunk = _slice_data_torchaudio(float(r["t0"]), float(r["t1"]), data, a.sample_rate)
        chunk = cut_pad_sample_torchaudio(chunk, a)
        golden.append({"sample_id": r["sample_id"], "wav": chunk.squeeze(0)})
    torch.save(golden, out / "pipeline_golden.pt")
    print("pipeline golden", len(golden), [g["sample_id"] for g in golden[:2]])
    print("완료:", out)


if __name__ == "__main__":
    main()
