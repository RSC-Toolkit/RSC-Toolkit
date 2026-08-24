"""
Patch-Mix 공식 데이터 파이프라인(cycle 추출→cut/pad→fbank, SpecAugment)에서
P0 골든을 뜬다.

모델 골든과 달리 이 스크립트는 **툴킷 venv 에서** 돌린다. 원 저장소가 torchaudio
버전을 고정하지 않았으므로(재현 위험 항목), 여기서의 기준은 라이브러리 버전이
아니라 저자 코드 자체다 — 같은 torchaudio 위에서 저자 코드와 우리 코드를 비교하면
코드 차이만 격리된다.

  .venv/bin/python scripts/reproduction/dump_patchmix_pipeline_goldens.py \
      --repo <patch-mix 클론> --data sample_data/toy_dataset/files

저자 util 모듈이 최상위에서 import 하는 cv2/cmapy/nlpaug 는 골든 경로에서 쓰이지
않으므로 스텁으로 대체한다 (get_individual_cycles_torchaudio / generate_fbank /
SpecAugment 는 torchaudio·librosa·numpy·torchvision 만 쓴다).
"""
import argparse
import json
import pathlib
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def install_stubs():
    _stub("cv2")
    _stub("cmapy")
    nlp = _stub("nlpaug")
    aug = _stub("nlpaug.augmenter")
    audio = _stub("nlpaug.augmenter.audio")
    nlp.augmenter = aug
    aug.audio = audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default="sample_data/toy_dataset/files")
    ap.add_argument("--out", default="tests/golden/patchmix_pipeline")
    ap.add_argument("--n_files", type=int, default=2)
    ap.add_argument("--n_cycles", type=int, default=2)
    args = ap.parse_args()

    install_stubs()
    sys.path.insert(0, str(pathlib.Path(args.repo).resolve()))

    from util.icbhi_util import get_annotations, get_individual_cycles_torchaudio, generate_fbank  # noqa: E402
    from util.augmentation import SpecAugment  # noqa: E402

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_folder = str(pathlib.Path(args.data).resolve())

    ref_args = SimpleNamespace(
        sample_rate=16000, desired_length=8, pad_types="repeat",
        class_split="lungsound", n_cls=4, butterworth_filter=None,
        specaug_policy="icbhi_ast_sup", specaug_mask="mean",
    )

    ann_dict = get_annotations(ref_args, data_folder)
    filenames = sorted(ann_dict.keys())[: args.n_files]

    entries = []
    fbanks = []
    for fn in filenames:
        ann = ann_dict[fn]
        cycles = get_individual_cycles_torchaudio(
            ref_args, ann, None, data_folder, fn, ref_args.sample_rate, ref_args.n_cls
        )
        rows = []
        for i in ann.index[: args.n_cycles]:
            rows.append({
                "start": float(ann.loc[i]["Start"]),
                "end": float(ann.loc[i]["End"]),
                "crackles": int(ann.loc[i]["Crackles"]),
                "wheezes": int(ann.loc[i]["Wheezes"]),
            })
        for k in range(min(args.n_cycles, len(cycles))):
            audio, label = cycles[k]
            fb = generate_fbank(audio, ref_args.sample_rate, n_mels=128)  # (798,128,1) numpy
            fbanks.append(torch.from_numpy(fb).squeeze(-1))
            rows[k]["label"] = int(label)
            rows[k]["audio_fingerprint"] = {
                "shape": list(audio.shape),
                "mean": float(audio.mean()), "std": float(audio.std()),
                "tail_mean_abs": float(audio[..., -1000:].abs().mean()),  # fade_out 검증용
            }
        entries.append({"filename": fn, "cycles": rows[: args.n_cycles]})

    torch.save(torch.stack(fbanks), out / "fbank_golden.pt")
    json.dump({"data_folder_note": "sample_data/toy_dataset/files", "entries": entries,
               "torch": torch.__version__}, open(out / "cycles_meta.json", "w"), indent=1)

    # ---- SpecAugment 골든 (같은 정책·같은 seed) ----
    img = fbanks[0].unsqueeze(0).clone()  # [1, 798, 128] = ToTensor((798,128,1)) 과 동일
    torch.save(img, out / "specaug_input.pt")
    torch.manual_seed(5); np.random.seed(5); random.seed(5)
    sa = SpecAugment(ref_args)
    ref_out = sa(img.clone())  # 참조는 in-place — clone 을 넘긴다
    assert not torch.equal(ref_out, img), "이 seed 에서 gate 가 스킵됨 — seed 를 바꿔라"
    torch.save(ref_out, out / "specaug_golden.pt")

    print("entries:", [(e["filename"], len(e["cycles"])) for e in entries])
    print("fbank golden:", tuple(torch.stack(fbanks).shape))
    print("완료:", out)


if __name__ == "__main__":
    main()
