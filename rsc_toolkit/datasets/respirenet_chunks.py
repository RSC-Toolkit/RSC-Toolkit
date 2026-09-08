"""
RespireNet(EMBC 2021)의 데이터 구성 — 공식 image_dataloader.py / utils.py 충실 이식 (P0).

원본 그대로: 녹음 txt 주석 읽기 → librosa.load(sr=4000) → cycle 슬라이스 → 8초 청킹
(>8s: 50% 겹침 프레임 + 꼬리 패딩, ≤8s: 자기 반복 패딩, 앞/뒤 정렬은 random) → 스펙트로그램 이미지.
identifier = f"{file}_{cycle_idx}_{label}" 로 청크→cycle 다수결 집계.

원본과 다른 점 (수치 영향 없음, 재현성 확보):
- 파일 순서를 os.listdir 대신 정렬 (원본은 파일시스템 순서).
- 패딩 방향 난수를 seed 있는 random.Random 으로 (원본은 무시드 전역 random → run 마다 특징이 달라짐).
- 리샘플 res_type 을 인자로 (기본 kaiser_best = 저자 환경 librosa 0.8 기본값; librosa 0.11 기본은 soxr_hq).
스플릿: 환자 fold 파일(80/20, test_fold) 또는 ICBHI 공식 60/40 (official_split.txt, 원본에는 없는 경로 — 툴킷 확장).
CBA(concatenation-based augmentation, 원본 new_augment) — 2026-09-08 사용자 결정으로 이식 (concat_augment).
  배포 코드는 (a) range(float) TypeError, (b) 이어 붙인 파일명 "A-B" 의 장치 표 KeyError 로 실행 불가였다(이슈 #1).
  (a) 는 int 강제, (b) 는 장치 표가 stage-1 학습에 쓰이지 않으므로(출력·stetho 필터 전용) 생략.
  scale 값은 논문·README·스크립트 어디에도 없어 **가정**(scale=1: 비정상 3클래스를 normal 수까지 증식, normal 증식 0).
  RNG 호출 순서(random()/randint)를 원본과 동일하게 유지 → tests/unit/datasets/test_respirenet_cba.py 에서 원본 함수와 동치 검증.
"""
from __future__ import annotations

import logging
import os
import pathlib
import random
from typing import List, Optional, Sequence, Tuple

import librosa
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rsc_toolkit.processors.respirenet_image import DEFAULT_MEAN, DEFAULT_STD, RespireNetImageProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 원본 utils.py
def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind:end_ind]


def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3


def generate_padded_samples(original, source, output_length, rng: random.Random):
    """원본 utils.py:156-179 (types=1: aug=original). prob 는 rng.random()."""
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length
    prob = rng.random()
    aug = original
    while len(aug) < left:
        aug = np.concatenate([aug, aug])
    if prob < 0.5:
        copy[left:] = source
        copy[:left] = aug[len(aug) - left:]
    else:
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]
    return copy


def split_and_pad(soundclip: np.ndarray, desired_length: float, sample_rate: int, rng: random.Random
                  ) -> List[Tuple[np.ndarray, int, float]]:
    """원본 utils.py:98-123 (types=1). 반환 [(chunk, chunk_idx, pad_times), ...]."""
    output_buffer_length = int(desired_length * sample_rate)
    soundclip = soundclip.copy()
    n_samples = len(soundclip)
    output = []
    if n_samples > output_buffer_length:
        frames = librosa.util.frame(soundclip, frame_length=output_buffer_length,
                                    hop_length=output_buffer_length // 2, axis=0)
        for i in range(frames.shape[0]):
            output.append((frames[i], i, 0))
        last_id = frames.shape[0] * (output_buffer_length // 2)
        last_sample = soundclip[last_id:]
        pad_times = (output_buffer_length - len(last_sample)) / len(last_sample)
        padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, rng)
        output.append((padded, i + 1, pad_times))
    else:
        padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, rng)
        pad_times = (output_buffer_length - len(soundclip)) / len(soundclip)
        output.append((padded, 0, pad_times))
    return output


def rollAudio(audio: np.ndarray, np_rng: np.random.RandomState) -> np.ndarray:
    pivot = np_rng.randint(audio.shape[0])
    return np.roll(audio, pivot, axis=0)


# ---------------------------------------------------------------- CBA (원본 image_dataloader.py:124-236 new_augment)
_CBA_PAIRS = {  # 클래스별 (누적 확률 상한, (첫 사이클 클래스, 둘째 사이클 클래스)) — 원본 하드코딩 그대로
    1: [(0.6, (1, 1)), (0.8, (1, 0)), (1.0, (0, 1))],
    2: [(0.6, (2, 2)), (0.8, (2, 0)), (1.0, (0, 2))],
    3: [(0.5, (3, 3)), (0.7, (1, 2)), (0.8, (2, 1)), (0.9, (3, 0)), (1.0, (0, 3))],
}


def concat_augment(classwise: Sequence[Sequence[tuple]], scale: int, rng: random.Random) -> List[tuple]:
    """원본 new_augment(scale) 의 충실 이식. classwise[c] = [(audio, label, file_name, cycle_idx), ...].
    클래스 c 를 scale*len(normal) 개까지 두 사이클 연결로 증식(부족분만; 음수면 0). 반환 [(audio, c, "A-B", idx), ...].
    RNG 호출 순서를 원본(전역 random)과 동일하게: normal 은 randint×2, 나머지는 random() → randint(i) → randint(j)."""
    scale = int(scale)
    out: List[tuple] = []
    n_normal = len(classwise[0])
    for idx in range(scale * n_normal - n_normal):                       # normal_i + normal_j
        i = rng.randint(0, n_normal - 1); j = rng.randint(0, n_normal - 1)
        a, b = classwise[0][i], classwise[0][j]
        out.append((np.concatenate([a[0], b[0]]), 0, a[2] + "-" + b[2], idx))
    for c in (1, 2, 3):
        for idx in range(scale * n_normal - len(classwise[c])):
            p = rng.random()
            ca, cb = next(pair for hi, pair in _CBA_PAIRS[c] if p < hi)
            i = rng.randint(0, len(classwise[ca]) - 1); j = rng.randint(0, len(classwise[cb]) - 1)
            a, b = classwise[ca][i], classwise[cb][j]
            out.append((np.concatenate([a[0], b[0]]), c, a[2] + "-" + b[2], idx))
    return out


# ---------------------------------------------------------------- 스플릿
def read_patient_folds(folds_file) -> dict:
    out = {}
    for line in pathlib.Path(folds_file).read_text().splitlines():
        line = line.strip()
        if line:
            pid, fold = line.split()
            out[pid] = int(fold)
    return out


def read_official_split(split_file) -> dict:
    out = {}
    for line in pathlib.Path(split_file).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1].lower()
    return out


# ---------------------------------------------------------------- 데이터셋
class RespireNetChunkDataset(Dataset):
    def __init__(
        self,
        data_dir,
        train_flag: bool,
        split_mode: str = "fold",              # "fold" (환자 5-fold, 원본) | "official" (ICBHI 60/40)
        folds_file: Optional[str] = None,
        test_fold: int = 4,
        official_split_file: Optional[str] = None,
        sample_rate: int = 4000,
        desired_length: float = 8.0,
        res_type: str = "kaiser_best",
        pad_seed: int = 0,
        augment: bool = False,                 # 학습 증강 (nlpaug 5종 + roll) — 2단계에서 연결
        aug_seed: int = 0,
        mean: Optional[Sequence[float]] = DEFAULT_MEAN,
        std: Optional[Sequence[float]] = DEFAULT_STD,
        stetho_id: int = -1,
        aug_scale: Optional[float] = None,     # CBA 배수 (원본 --aug_scale; None/0 = 끔). 학습셋에만.
        cba_seed: Optional[int] = None,        # CBA 표본 RNG seed (None → aug_seed). 원본은 무시드 전역 random.
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.train_flag = bool(train_flag)
        self.sample_rate, self.desired_length, self.res_type = int(sample_rate), float(desired_length), res_type
        self.augment = bool(augment) and self.train_flag
        self.processor = RespireNetImageProcessor(sampling_rate=sample_rate, mean=mean or DEFAULT_MEAN, std=std or DEFAULT_STD)

        # 녹음 파일만 (툴킷 데이터 디렉토리에는 filename_format.txt 같은 부속 txt 도 있다 — 원본 디렉토리엔 없음)
        filenames = sorted(s[:-4] for s in os.listdir(self.data_dir)
                           if s.endswith(".txt") and len(s[:-4].split("_")) == 5 and (self.data_dir / (s[:-4] + ".wav")).exists())
        # 장치 id: 원본은 os.listdir 등장 순 — 저자 로그 순서를 고정값으로 쓴다
        self.device_to_id = {"AKGC417L": 0, "Meditron": 1, "LittC2SE": 2, "Litt3200": 3}
        self.file_to_device = {f: self.device_to_id[f.split("_")[-1]] for f in filenames}

        if split_mode == "fold":
            folds = read_patient_folds(folds_file)
            keep = lambda f: (folds[f.split("_")[0]] == test_fold) != self.train_flag  # noqa: E731
        elif split_mode == "official":
            sp = read_official_split(official_split_file)
            keep = lambda f: sp.get(f) == ("train" if self.train_flag else "test")  # noqa: E731
        else:
            raise ValueError(f"split_mode must be fold|official, got {split_mode!r}")
        self.filenames = [f for f in filenames if keep(f) and (stetho_id < 0 or self.file_to_device[f] == stetho_id)]

        # 1) cycle 추출 (원본 cycle_list / classwise_cycle_list)
        cycles: List[tuple] = []                       # (audio, label, file_name, cycle_idx)
        classwise: List[List[tuple]] = [[], [], [], []]
        for file_name in self.filenames:
            ann = pd.read_csv(self.data_dir / f"{file_name}.txt", names=["Start", "End", "Crackles", "Wheezes"], delimiter="\t")
            data, _ = librosa.load(str(self.data_dir / f"{file_name}.wav"), sr=self.sample_rate, res_type=self.res_type)
            for cycle_idx in range(len(ann.index)):
                row = ann.loc[cycle_idx]
                chunk = slice_data(row["Start"], row["End"], data, self.sample_rate)
                label = get_label(row["Crackles"], row["Wheezes"])
                cycles.append((chunk, label, file_name, cycle_idx))
                classwise[label].append(cycles[-1])
        self.n_cycles_raw = len(cycles)
        self.n_cycles_cba = 0
        # 2) CBA — 원본과 같이 cycle 추출 직후·청킹 전, 학습셋만 (test 는 절대 안 함)
        self.aug_scale = float(aug_scale) if (aug_scale and self.train_flag) else 0.0
        if self.aug_scale:
            cba = concat_augment(classwise, self.aug_scale, random.Random(aug_seed if cba_seed is None else cba_seed))
            cycles.extend(cba)
            self.n_cycles_cba = len(cba)
        # 3) 8초 청킹 (원본 split_and_pad; 패딩 방향 RNG = pad_seed) — 원본 cycle 순서 그대로, CBA 사이클은 뒤에
        rng = random.Random(pad_seed)
        self.audio_data: List[Tuple[np.ndarray, int, str, int, int, float]] = []  # chunk,label,file,cycle_idx,chunk_idx,pad_times
        self.identifiers: List[str] = []
        self.cycle_ids: List[str] = []       # cycle 단위 identifier (다수결 대상)
        for chunk, label, file_name, cycle_idx in cycles:
            cid = f"{file_name}_{cycle_idx}_{label}"
            self.cycle_ids.append(cid)
            for padded, chunk_idx, pad_times in split_and_pad(chunk, self.desired_length, self.sample_rate, rng):
                self.audio_data.append((padded, label, file_name, cycle_idx, chunk_idx, pad_times))
                self.identifiers.append(cid)
        self.labels = [s[1] for s in self.audio_data]
        counts = np.bincount(self.labels, minlength=4).astype(np.float64)
        self.class_probs = counts / counts.sum()
        self._np_rng = np.random.RandomState(aug_seed)
        self._py_rng = random.Random(aug_seed + 1)
        logger.info("RespireNet %s: 녹음 %d, cycle %d (원본 %d + CBA %d, scale=%s), 청크 %d, 클래스 %s",
                    "train" if train_flag else "test", len(self.filenames), len(self.cycle_ids), self.n_cycles_raw,
                    self.n_cycles_cba, self.aug_scale or "off", len(self.audio_data), counts.astype(int).tolist())

    def __len__(self):
        return len(self.audio_data)

    def chunk_audio(self, index: int) -> np.ndarray:
        audio = self.audio_data[index][0]
        if self.augment:
            from rsc_toolkit.augmentations.respirenet_raw import gen_augmented  # 2단계

            if self._py_rng.random() > 0.5:
                audio = gen_augmented(audio, self.sample_rate, self._np_rng, self._py_rng)
                audio = split_and_pad(audio, self.desired_length, self.sample_rate, self._py_rng)[0][0]
            if self._py_rng.random() > 0.5:
                audio = rollAudio(audio, self._np_rng)
        return audio

    def set_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.processor.mean, self.processor.std = tuple(float(v) for v in mean), tuple(float(v) for v in std)

    def raw_image_tensor(self, index: int) -> torch.Tensor:
        """ToTensor 만 적용한 (3, H, W) — 원본 get_mean_and_std 입력."""
        img = self.processor.image(self.chunk_audio(index))
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, index: int):
        audio = self.chunk_audio(index)
        return self.processor.make_features(audio), int(self.audio_data[index][1]), index


def respirenet_collate(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    idx = torch.as_tensor([b[2] for b in batch], dtype=torch.long)
    return x, y, idx


class RespireNetDataModule(L.LightningDataModule):
    meta_kind = "respirenet_chunk_index"  # batch[2] = 청크 인덱스 (사이클 투표 콜백용) — respirenet 모델은 무시

    def __init__(self, data_dir: str = ".datasets/icbhi", split_mode: str = "fold",
                 folds_file: str = "rsc_toolkit/datasets/icbhi/data/patient_list_foldwise.txt", test_fold: int = 4,
                 official_split_file: str = "rsc_toolkit/datasets/icbhi/data/official_split.txt",
                 sample_rate: int = 4000, desired_length: float = 8.0, res_type: str = "kaiser_best",
                 pad_seed: int = 0, augment: bool = True, aug_seed: int = 0,
                 mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None,
                 batch_size: int = 64, num_workers: int = 4, weighted_sampler: bool = True, stetho_id: int = -1,
                 aug_scale: Optional[float] = None, cba_seed: Optional[int] = None):
        super().__init__()
        self.kw = dict(data_dir=data_dir, split_mode=split_mode, folds_file=folds_file, test_fold=test_fold,
                       official_split_file=official_split_file, sample_rate=sample_rate, desired_length=desired_length,
                       res_type=res_type, pad_seed=pad_seed, aug_seed=aug_seed, mean=None, std=None,
                       stetho_id=stetho_id, aug_scale=aug_scale, cba_seed=cba_seed)
        self.augment = bool(augment)
        self.batch_size, self.num_workers, self.weighted_sampler = int(batch_size), int(num_workers), bool(weighted_sampler)
        self.mean = tuple(mean) if mean is not None else None
        self.std = tuple(std) if std is not None else None
        self.train_ds = self.test_ds = None

    def _compute_mean_std(self, ds) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """원본 utils.get_mean_and_std — 증강이 켜진 학습셋의 이미지별 채널 mean/std 를 평균."""
        class _Stats(Dataset):
            def __len__(_s):
                return len(ds)

            def __getitem__(_s, i):
                x = ds.raw_image_tensor(i)
                return x.mean(dim=(1, 2)), x.std(dim=(1, 2))

        mean = torch.zeros(3); std = torch.zeros(3)
        for m, sd in DataLoader(_Stats(), batch_size=32, num_workers=self.num_workers):
            mean += m.sum(0); std += sd.sum(0)
        mean /= len(ds); std /= len(ds)
        logger.info("RespireNet 정규화 통계 (train, 증강 포함): mean %s std %s", mean.tolist(), std.tolist())
        return tuple(mean.tolist()), tuple(std.tolist())

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_ds is None:
            self.train_ds = RespireNetChunkDataset(train_flag=True, augment=self.augment, **self.kw)
            if self.mean is None or self.std is None:
                self.mean, self.std = self._compute_mean_std(self.train_ds)
            self.train_ds.set_normalization(self.mean, self.std)
        if self.test_ds is None:
            self.test_ds = RespireNetChunkDataset(train_flag=False, augment=False, **self.kw)
            if self.mean is None or self.std is None:  # 평가 전용 (fit 없이 test): eval.py 상수
                self.mean, self.std = DEFAULT_MEAN, DEFAULT_STD
            self.test_ds.set_normalization(self.mean, self.std)

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.weighted_sampler:
            # 원본 train.py:114-118 — 청크 라벨의 클래스 확률 역수
            w = 1.0 / torch.tensor([self.train_ds.class_probs[l] for l in self.train_ds.labels], dtype=torch.float32)
            sampler = torch.utils.data.WeightedRandomSampler(w, len(self.train_ds))
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, shuffle=sampler is None,
                          num_workers=self.num_workers, collate_fn=respirenet_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=respirenet_collate)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
