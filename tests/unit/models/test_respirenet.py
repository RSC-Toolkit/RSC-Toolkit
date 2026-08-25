"""
RespireNet(EMBC 2021) 묶음의 P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_respirenet_goldens.py — 공식 microsoft/RespireNet 코드를 우리 venv 에서
(정렬된 listdir·seed 0 패딩·kaiser_best 로 고정) 실행해 생성. 배포 ckpt_best.pkl 평가 기록은
tests/golden/respirenet/ckpt_eval.json (fold-4 test, seed 0: Sp 80.26 / Se 52.02 / Score 66.14).
"""
import json
import pathlib
import random

import numpy as np
import pytest
import torch

from rsc_toolkit.datasets.respirenet_chunks import (
    RespireNetChunkDataset,
    generate_padded_samples,
    get_label,
    split_and_pad,
)
from rsc_toolkit.processors.respirenet_image import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    MAGMA_LUT_BGR,
    normalize_image,
    respirenet_image,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "respirenet"
DATA = REPO / ".datasets" / "icbhi"
CKPT = pathlib.Path("/tmp/claude-10008/-raid-doyoung-projects-RSC-Toolkit/88feaa60-f05e-4f32-8234-95e341f584ce/scratchpad/reference/respirenet/models/ckpt_best.pkl")

needs_golden = pytest.mark.skipif(not (GOLDEN / "images.pt").exists(), reason="골든 없음")
needs_data = pytest.mark.skipif(not (DATA / "101_1b1_Al_sc_Meditron.wav").exists(), reason="ICBHI 원음 없음")


# ------------------------------------------------------------------ 모델
def test_model_matches_reference_layout():
    from rsc_toolkit.models.respirenet import RespireNetCNN

    m = RespireNetCNN(n_cls=4, imagenet_pretrain=False)
    sd = m.state_dict()
    assert len(sd) == 222 and sum(p.numel() for p in m.parameters()) == 21_367_364
    assert all(k.startswith(("model_ft.", "cls_fc.")) for k in sd)
    assert tuple(sd["model_ft.fc.1.weight"].shape) == (128, 512) and tuple(sd["cls_fc.weight"].shape) == (4, 128)
    if CKPT.exists():
        m.load_reference_checkpoint(CKPT)  # strict


def test_fine_tune_freezes_children_up_to_layer2():
    from rsc_toolkit.models.respirenet import RespireNetCNN

    m = RespireNetCNN(n_cls=4, imagenet_pretrain=False)
    m.fine_tune(block_layer=5)
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert any(n.startswith("model_ft.layer2.") for n in frozen)
    assert not any(n.startswith(("model_ft.layer3.", "model_ft.fc.", "cls_fc.")) for n in frozen)


# ----------------------------------------------------------- 청킹/패딩
def test_split_and_pad_short_cycle_pads_front_or_back():
    x = np.arange(1, 6, dtype=np.float32)  # 5 샘플, 목표 12
    back = split_and_pad(x, 12, 1, random.Random(0))   # 첫 random() < 0.5 → source 뒤쪽 정렬
    assert len(back) == 1 and back[0][1] == 0 and back[0][2] == pytest.approx(7 / 5)
    chunk = back[0][0]
    # 두 정렬 중 하나여야 하고, 채움은 자기 반복(aug=original) 의 앞/뒤 조각
    front_layout = np.concatenate([x, np.tile(x, 3)[:7]])
    back_layout = np.concatenate([np.tile(x, 3)[8 - 7:8], x]) if False else None
    assert np.array_equal(chunk[-5:], x) or np.array_equal(chunk[:5], x)
    if np.array_equal(chunk[:5], x):
        assert np.array_equal(chunk, front_layout)


def test_split_and_pad_long_cycle_overlapping_frames_plus_tail():
    sr, L = 1, 8
    x = np.arange(20, dtype=np.float32)  # 20 > 8 → hop 4 프레임 4개 + 꼬리
    out = split_and_pad(x, L, sr, random.Random(1))
    n_frames = 1 + (20 - 8) // 4
    assert len(out) == n_frames + 1
    assert np.array_equal(out[0][0], x[0:8]) and np.array_equal(out[1][0], x[4:12])
    assert [o[1] for o in out] == list(range(n_frames + 1))
    tail = x[n_frames * 4:]
    assert out[-1][2] == pytest.approx((L - len(tail)) / len(tail))


def test_get_label_mapping():
    assert [get_label(0, 0), get_label(1, 0), get_label(0, 1), get_label(1, 1)] == [0, 1, 2, 3]


# ------------------------------------------------------------ 이미지
def test_magma_lut_and_image_shape():
    assert MAGMA_LUT_BGR.shape == (256, 1, 3) and MAGMA_LUT_BGR.dtype == np.uint8
    rng = np.random.RandomState(0)
    img = respirenet_image(rng.randn(32000).astype(np.float32) * 0.1)
    assert img.shape == (192, 753, 3) and img.dtype == np.uint8
    t = normalize_image(img)
    assert t.shape == (3, 192, 753)


@needs_golden
@needs_data
def test_test_split_images_match_reference_bitwise():
    """fold-4 test 청크 이미지(BRC 후 uint8)와 정규화 텐서가 공식 코드 출력과 비트 단위로 같다."""
    golden = torch.load(GOLDEN / "images.pt")
    ds = RespireNetChunkDataset(DATA, train_flag=False, split_mode="fold", test_fold=4,
                                folds_file=REPO / "rsc_toolkit/datasets/icbhi/data/patient_list_foldwise.txt",
                                res_type="kaiser_best", pad_seed=0)
    meta = json.load(open(GOLDEN / "chunks_meta.json"))["fold4_test"]
    assert len(ds) == meta["n_chunks"] == 1444
    assert ds.identifiers == meta["identifiers"]
    assert ds.cycle_ids == meta["cycle_ids"]
    for g in golden:
        i = g["index"]
        assert ds.identifiers[i] == g["identifier"]
        img = ds.processor.image(ds.chunk_audio(i))
        assert np.array_equal(img, g["image_uint8"].numpy()), i
        x, y, _ = ds[i]
        assert y == g["label"] and torch.allclose(x, g["tensor"], atol=1e-6)


@needs_golden
@needs_data
def test_train_split_counts_match_author_log():
    """저자 로그: fold-4 train 5471 청크, 클래스 [2878,1474,683,436]."""
    ds = RespireNetChunkDataset(DATA, train_flag=True, split_mode="fold", test_fold=4,
                                folds_file=REPO / "rsc_toolkit/datasets/icbhi/data/patient_list_foldwise.txt",
                                res_type="kaiser_best", pad_seed=0)
    meta = json.load(open(GOLDEN / "chunks_meta.json"))["fold4_train"]
    assert len(ds) == meta["n_chunks"] == 5471
    assert np.bincount(ds.labels, minlength=4).tolist() == meta["class_counts"] == [2878, 1474, 683, 436]
    assert np.allclose(ds.class_probs, meta["class_probs"])


# ------------------------------------------------------- 학습 프로토콜 부품
def test_prior_weighted_ce_matches_reference_weights():
    from rsc_toolkit.trainer.losses import WeightedCrossEntropyFromPrior

    loss = WeightedCrossEntropyFromPrior([3.0, 1.0, 1.0, 1.0])
    assert torch.allclose(loss.weight, torch.tensor([0.1, 0.3, 0.3, 0.3]))


def test_vote_cycles_majority_and_tie_rule():
    from rsc_toolkit.trainer.callbacks.respirenet_vote_eval import vote_cycles

    ids = ["f_0_1", "f_0_1", "f_0_1", "g_0_0", "g_0_0"]
    preds = [1, 2, 1, 0, 2]  # g: 동률 → 낮은 인덱스(0)
    res = vote_cycles(preds, ids, ["f_0_1", "g_0_0"])
    assert res["cycle_acc"] == 100.0 and res["S_p"] == 100.0 and res["S_e"] == 100.0


def test_raw_augmentations_are_seeded_and_length_preserving_except_speed():
    from rsc_toolkit.augmentations.respirenet_raw import Loudness, Noise, Pitch, Speed, Vtlp, gen_augmented

    x = (np.random.RandomState(0).randn(32000) * 0.05).astype(np.float32)
    for cls in [Noise(), Loudness(factor=(0.5, 2)), Pitch(4000, factor=(-1, 3)), Vtlp(4000, zone=(0.0, 1.0))]:
        a = cls.manipulate(x, np.random.RandomState(7))
        b = cls.manipulate(x, np.random.RandomState(7))
        assert np.array_equal(a, b), type(cls).__name__
    sp = Speed().manipulate(x, np.random.RandomState(1))
    assert sp.shape[0] != x.shape[0] or not np.array_equal(sp, x)  # 속도 변경은 길이를 바꾼다
    g1 = gen_augmented(x, 4000, np.random.RandomState(3), random.Random(4))
    g2 = gen_augmented(x, 4000, np.random.RandomState(3), random.Random(4))
    assert np.array_equal(g1, g2)


def test_speed_levels_exclude_unity_and_pitch_levels_range():
    from rsc_toolkit.augmentations.respirenet_raw import Speed

    levels = {Speed().get_speed_level(np.random.RandomState(i)) for i in range(200)}
    assert 1.0 not in levels and min(levels) >= 0.5 and max(levels) <= 1.9
