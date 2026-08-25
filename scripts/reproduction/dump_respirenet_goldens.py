"""
RespireNet(microsoft/RespireNet) P0 골든 — 공식 코드를 우리 venv 에서 실행 (cv2·cmapy·nlpaug 는
스크래치 pylibs 를 PYTHONPATH 로).

  PYTHONPATH=<pylibs> .venv/bin/python scripts/reproduction/dump_respirenet_goldens.py \
      --repo <RespireNet 클론> --data .datasets/icbhi --out tests/golden/respirenet

원본을 재현 가능하게 만드는 패치(수치 분포는 동일): os.listdir → 정렬, random.random(패딩 방향) → seed 0
Random, librosa.load res_type=kaiser_best 고정. 골든:
- chunks_meta.json  : fold-4 test 의 identifiers/labels/청크 수, official 60/40 train/test 청크 수
- images.pt         : fold-4 test 처음 6 청크(+장치별 1개씩)의 BRC 후 uint8 RGB 이미지와 정규화 텐서
- ckpt_eval.json    : ckpt_best.pkl 로 fold-4 test 전수 평가 — 청크 예측, cycle 다수결 conf matrix, Sp/Se/Score
"""
import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default=".datasets/icbhi")
    ap.add_argument("--out", default="tests/golden/respirenet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    # ---- 재현성 패치 ----
    _listdir = os.listdir
    os.listdir = lambda p: sorted(_listdir(p))
    seeded = random.Random(0)
    random.random = seeded.random
    import librosa
    _load = librosa.load
    librosa.load = lambda *a, **k: _load(*a, **{**k, "res_type": "kaiser_best"})

    import image_dataloader  # noqa: E402
    from torchvision.transforms import Compose, Normalize, ToTensor  # noqa: E402
    mean, std = [0.5091, 0.1739, 0.4363], [0.2288, 0.1285, 0.0743]
    tf = Compose([ToTensor(), Normalize(mean, std)])
    folds = str(repo / "data" / "patient_list_foldwise.txt")

    # 원본 device_to_id 는 listdir 순 — 정렬하면 순서가 달라지므로 우리 고정값과 대조만 기록
    test = image_dataloader.image_loader(args.data, folds, 4, False, "params_json", tf, -1)
    meta = {"fold4_test": {"n_chunks": len(test), "identifiers": test.identifiers, "labels": [int(l) for l in test.labels],
                           "cycle_ids": test.filenames_with_labels}}
    train_rng_state = seeded.getstate()
    train = image_dataloader.image_loader(args.data, folds, 4, True, "params_json", tf, -1)
    meta["fold4_train"] = {"n_chunks": len(train), "class_counts": np.bincount(train.labels, minlength=4).tolist(),
                           "class_probs": train.class_probs.tolist()}
    json.dump(meta, open(out / "chunks_meta.json", "w"))
    print("fold4 test chunks", len(test), "train chunks", len(train), train.class_probs)

    # ---- 이미지 골든 (test 는 __getitem__ 에 난수 없음) ----
    dev_first = {}
    for i, s in enumerate(test.audio_data):
        d = test.file_to_device[s[2]]
        dev_first.setdefault(d, i)
    idxs = sorted(set(list(range(6)) + list(dev_first.values())))
    images = []
    for i in idxs:
        # 원본 __getitem__ 의 이미지 단계만 (transform 전 uint8) 재현하기 위해 transform 을 잠시 끈다
        test.input_transform = None
        img_uint8, label = test[i]
        test.input_transform = tf
        x, _ = test[i]
        images.append({"index": i, "identifier": test.identifiers[i], "label": int(label),
                       "image_uint8": torch.from_numpy(np.ascontiguousarray(img_uint8)), "tensor": x})
    torch.save(images, out / "images.pt")
    print("이미지 골든", [(im["index"], tuple(im["image_uint8"].shape)) for im in images])

    # ---- ckpt 평가 골든 ----
    from nets.network_cnn import model as Model  # noqa: E402
    net = Model(num_classes=4)
    net.load_state_dict(torch.load(str(repo / "models" / "ckpt_best.pkl"), map_location="cpu", weights_only=True))
    net = net.to(args.device).eval()
    loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(net(x.to(args.device)).argmax(1).cpu().tolist())
    preds = np.array(preds); ids = np.array(test.identifiers)
    y_pred, y_true = [], []
    for pt in test.filenames_with_labels:
        y_pred.append(int(np.argmax(np.bincount(preds[np.where(ids == pt)]))))
        y_true.append(int(pt.split("_")[-1]))
    conf = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    sp = conf[0, 0] / conf[0].sum() * 100
    se = np.trace(conf[1:, 1:]) / conf[1:].sum() * 100
    rec = {"chunk_preds": preds.tolist(), "conf": conf.tolist(), "S_p": sp, "S_e": se, "icbhi_score": (sp + se) / 2,
           "acc": float((np.array(y_true) == np.array(y_pred)).mean() * 100), "pad_seed": 0, "res_type": "kaiser_best"}
    json.dump(rec, open(out / "ckpt_eval.json", "w"), indent=1)
    print(f"ckpt eval: Sp {sp:.2f} / Se {se:.2f} / Score {(sp+se)/2:.2f} acc {rec['acc']:.2f}")
    print("완료:", out)


if __name__ == "__main__":
    main()
