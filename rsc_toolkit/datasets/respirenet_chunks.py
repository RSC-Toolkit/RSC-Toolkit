# Portions of this file are adapted from RespireNet (https://github.com/microsoft/RespireNet); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Data pipeline of RespireNet (EMBC 2021) -- exact port of the official image_dataloader.py / utils.py (verified against goldens).

As in the original: read recording txt annotations -> librosa.load(sr=4000) -> slice cycles -> 8 s chunking
(>8 s: 50%-overlap frames + tail padding, <=8 s: self-repeat padding, front/back alignment random) -> spectrogram image.
Chunk -> cycle majority vote via identifier = f"{file}_{cycle_idx}_{label}".

Differences from the original (no numeric impact, for reproducibility):
- File order is sorted instead of os.listdir (the original uses filesystem order).
- The padding-direction RNG is a seeded random.Random (the original uses the unseeded global random -> features differ per run).
- The resampling res_type is an argument (default kaiser_best = the librosa 0.8 default in the authors' environment; librosa 0.11 defaults to soxr_hq).
Split: patient fold files (80/20, test_fold) or the official ICBHI 60/40 split (official_split.txt; not in the original code -- toolkit extension).
CBA (concatenation-based augmentation, original new_augment) is ported (concat_augment).
  The released code cannot run due to (a) a range(float) TypeError and (b) a device-table KeyError for the concatenated file name "A-B".
  (a) is fixed by casting to int; (b) is skipped because the device table is not used in stage-1 training (only for output and the stetho filter).
  The scale value appears nowhere in the paper, README or scripts, so it is **assumed** (scale=1: the 3 abnormal classes are grown up to the normal count, normal not grown).
  The RNG call order (random()/randint) is kept identical to the original -> equivalence with the original function is tested in tests/unit/datasets/test_respirenet_cba.py.
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


# ---------------------------------------------------------------- original utils.py
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
    """Original utils.py:156-179 (types=1: aug=original). prob is rng.random()."""
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
    """Original utils.py:98-123 (types=1). Returns [(chunk, chunk_idx, pad_times), ...]."""
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


# ---------------------------------------------------------------- CBA (original image_dataloader.py:124-236 new_augment)
_CBA_PAIRS = {  # per class (cumulative probability upper bound, (first cycle class, second cycle class)) -- hard-coded as in the original
    1: [(0.6, (1, 1)), (0.8, (1, 0)), (1.0, (0, 1))],
    2: [(0.6, (2, 2)), (0.8, (2, 0)), (1.0, (0, 2))],
    3: [(0.5, (3, 3)), (0.7, (1, 2)), (0.8, (2, 1)), (0.9, (3, 0)), (1.0, (0, 3))],
}


def concat_augment(classwise: Sequence[Sequence[tuple]], scale: int, rng: random.Random) -> List[tuple]:
    """Exact port of the original new_augment(scale). classwise[c] = [(audio, label, file_name, cycle_idx), ...].
    Grows class c up to scale*len(normal) by concatenating two cycles (only the shortfall; 0 if negative). Returns [(audio, c, "A-B", idx), ...].
    RNG call order identical to the original (global random): normal uses randint x2, others random() -> randint(i) -> randint(j)."""
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


# ---------------------------------------------------------------- split
def read_patient_folds(folds_file) -> dict:
    out = {}
    for line in pathlib.Path(folds_file).read_text().splitlines():
        line = line.strip()
        if line:
            pid, fold = line.split()
            out[pid] = int(fold)
    return out


from rsc_toolkit.datasets.splits import read_split_file as read_official_split  # noqa: E402  single source of the parser


# ---------------------------------------------------------------- dataset
class RespireNetChunkDataset(Dataset):
    def __init__(
        self,
        data_dir,
        train_flag: bool,
        split_mode: str = "fold",              # "fold" (patient 5-fold, original) | "official" (ICBHI 60/40)
        folds_file: Optional[str] = None,
        test_fold: int = 4,
        official_split_file: Optional[str] = None,
        sample_rate: int = 4000,
        desired_length: float = 8.0,
        res_type: str = "kaiser_best",
        pad_seed: int = 0,
        augment: bool = False,                 # training augmentation (5 nlpaug ops + roll) -- wired in stage 2
        aug_seed: int = 0,
        mean: Optional[Sequence[float]] = DEFAULT_MEAN,
        std: Optional[Sequence[float]] = DEFAULT_STD,
        stetho_id: int = -1,
        aug_scale: Optional[float] = None,     # CBA multiplier (original --aug_scale; None/0 = off). Training set only.
        cba_seed: Optional[int] = None,        # CBA sampling RNG seed (None -> aug_seed). The original uses the unseeded global random.
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.train_flag = bool(train_flag)
        self.sample_rate, self.desired_length, self.res_type = int(sample_rate), float(desired_length), res_type
        self.augment = bool(augment) and self.train_flag
        self.processor = RespireNetImageProcessor(sampling_rate=sample_rate, mean=mean or DEFAULT_MEAN, std=std or DEFAULT_STD)

        # recording files only (the toolkit data directory also contains auxiliary txt files such as filename_format.txt -- the original directory does not)
        filenames = sorted(s[:-4] for s in os.listdir(self.data_dir)
                           if s.endswith(".txt") and len(s[:-4].split("_")) == 5 and (self.data_dir / (s[:-4] + ".wav")).exists())
        # device ids: the original uses os.listdir order -- the order from the authors' logs is used as a fixed value
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

        # 1) cycle extraction (original cycle_list / classwise_cycle_list)
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
        # 2) CBA -- as in the original, right after cycle extraction and before chunking, training set only (never on test)
        self.aug_scale = float(aug_scale) if (aug_scale and self.train_flag) else 0.0
        if self.aug_scale:
            cba = concat_augment(classwise, self.aug_scale, random.Random(aug_seed if cba_seed is None else cba_seed))
            cycles.extend(cba)
            self.n_cycles_cba = len(cba)
        # 3) 8 s chunking (original split_and_pad; padding-direction RNG = pad_seed) -- original cycle order kept, CBA cycles appended
        rng = random.Random(pad_seed)
        self.audio_data: List[Tuple[np.ndarray, int, str, int, int, float]] = []  # chunk,label,file,cycle_idx,chunk_idx,pad_times
        self.identifiers: List[str] = []
        self.cycle_ids: List[str] = []       # cycle-level identifier (majority vote unit)
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
        logger.info("RespireNet %s: recordings %d, cycles %d (original %d + CBA %d, scale=%s), chunks %d, classes %s",
                    "train" if train_flag else "test", len(self.filenames), len(self.cycle_ids), self.n_cycles_raw,
                    self.n_cycles_cba, self.aug_scale or "off", len(self.audio_data), counts.astype(int).tolist())

    def __len__(self):
        return len(self.audio_data)

    def chunk_audio(self, index: int) -> np.ndarray:
        audio = self.audio_data[index][0]
        if self.augment:
            from rsc_toolkit.augmentations.respirenet_raw import gen_augmented  # stage 2

            if self._py_rng.random() > 0.5:
                audio = gen_augmented(audio, self.sample_rate, self._np_rng, self._py_rng)
                audio = split_and_pad(audio, self.desired_length, self.sample_rate, self._py_rng)[0][0]
            if self._py_rng.random() > 0.5:
                audio = rollAudio(audio, self._np_rng)
        return audio

    def set_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.processor.mean, self.processor.std = tuple(float(v) for v in mean), tuple(float(v) for v in std)

    def raw_image_tensor(self, index: int) -> torch.Tensor:
        """(3, H, W) with only ToTensor applied -- input of the original get_mean_and_std."""
        img = self.processor.image(self.chunk_audio(index))
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, index: int):
        audio = self.chunk_audio(index)
        return self.processor.make_features(audio), int(self.audio_data[index][1]), index



class RespireNetIndexChunkDataset(Dataset):
    """**Evaluation-only** dataset that reads cycles from a toolkit index (index_cycle*.csv) and turns them into RespireNet chunks (4 kHz, 8 s repeat/pad, magma image).

    Path for zero-shot evaluation of a RespireNet ckpt trained on ICBHI on the SPRSound 2022 inter-patient test.
    The preprocessing chain is identical to the test path of RespireNetChunkDataset (no augmentation or CBA; split_and_pad padding RNG is pad_seed).
    Labels come straight from the index csv label column (for SPRSound, pass the 7->4 merged derived index index_cycle_icbhi4.csv).
    Cycle ids have the same "<file>_<cycle_idx>_<label>" form as the original, so the cycle majority vote of RespireNetVoteEval works unchanged.
    """

    def __init__(self, index_csv, data_dir, split: str = "test", sample_rate: int = 4000, desired_length: float = 8.0,
                 res_type: str = "kaiser_best", pad_seed: int = 0, mean: Optional[Sequence[float]] = DEFAULT_MEAN,
                 std: Optional[Sequence[float]] = DEFAULT_STD, n_classes: int = 4):
        self.data_dir = pathlib.Path(data_dir)
        self.train_flag = False
        self.augment = False
        self.sample_rate, self.desired_length, self.res_type = int(sample_rate), float(desired_length), res_type
        self.processor = RespireNetImageProcessor(sampling_rate=sample_rate, mean=mean or DEFAULT_MEAN, std=std or DEFAULT_STD)
        df = pd.read_csv(index_csv)
        df = df[df["split"] == split] if "split" in df.columns else df
        cycles: List[tuple] = []
        for rec, grp in df.groupby("recording_path", sort=True):
            data, _ = librosa.load(str(self.data_dir / rec), sr=self.sample_rate, res_type=self.res_type)
            stem = pathlib.Path(rec).stem
            for cycle_idx, (_, row) in enumerate(grp.sort_values(["t0", "t1"]).iterrows()):
                chunk = slice_data(float(row["t0"]), float(row["t1"]), data, self.sample_rate)
                cycles.append((chunk, int(row["label"]), stem, cycle_idx))
        self.filenames = sorted(df["recording_path"].unique().tolist())
        self.n_cycles_raw, self.n_cycles_cba, self.aug_scale = len(cycles), 0, 0.0
        rng = random.Random(pad_seed)
        self.audio_data: List[Tuple[np.ndarray, int, str, int, int, float]] = []
        self.identifiers: List[str] = []
        self.cycle_ids: List[str] = []
        for chunk, label, file_name, cycle_idx in cycles:
            cid = f"{file_name}_{cycle_idx}_{label}"
            self.cycle_ids.append(cid)
            for padded, chunk_idx, pad_times in split_and_pad(chunk, self.desired_length, self.sample_rate, rng):
                self.audio_data.append((padded, label, file_name, cycle_idx, chunk_idx, pad_times))
                self.identifiers.append(cid)
        self.labels = [s[1] for s in self.audio_data]
        counts = np.bincount(self.labels, minlength=n_classes).astype(np.float64)
        self.class_probs = counts / counts.sum()
        self.cycle_label_counts = np.bincount([int(c.split("_")[-1]) for c in self.cycle_ids], minlength=n_classes).tolist()
        logger.info("RespireNet index(%s): recordings %d, cycles %d, chunks %d, cycle classes %s", index_csv, len(self.filenames),
                    len(self.cycle_ids), len(self.audio_data), self.cycle_label_counts)

    def __len__(self):
        return len(self.audio_data)

    def chunk_audio(self, index: int) -> np.ndarray:
        return self.audio_data[index][0]

    def set_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.processor.mean, self.processor.std = tuple(float(v) for v in mean), tuple(float(v) for v in std)

    def raw_image_tensor(self, index: int) -> torch.Tensor:
        img = self.processor.image(self.chunk_audio(index))
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, index: int):
        audio = self.chunk_audio(index)
        return self.processor.make_features(audio), int(self.audio_data[index][1]), index

from rsc_toolkit.datasets import collate as _collate  # single collate implementation

# (x, y, chunk index long)
respirenet_collate = _collate.index_meta_collate


class RespireNetDataModule(L.LightningDataModule):
    meta_kind = "respirenet_chunk_index"  # batch[2] = chunk index (for the cycle-voting callback) -- ignored by the respirenet model

    def __init__(self, data_dir: str = ".datasets/icbhi", split_mode: str = "fold",
                 folds_file: str = "rsc_toolkit/datasets/icbhi/data/patient_list_foldwise.txt", test_fold: int = 4,
                 official_split_file: str = "rsc_toolkit/datasets/icbhi/data/official_split.txt",
                 sample_rate: int = 4000, desired_length: float = 8.0, res_type: str = "kaiser_best",
                 pad_seed: int = 0, augment: bool = True, aug_seed: int = 0,
                 mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None,
                 batch_size: int = 64, num_workers: int = 4, weighted_sampler: bool = True, stetho_id: int = -1,
                 aug_scale: Optional[float] = None, cba_seed: Optional[int] = None,
                 ood_index_csv: Optional[str] = None, ood_data_dir: Optional[str] = None, ood_split: str = "test"):
        super().__init__()
        # Zero-shot (OOD) evaluation: only the test set is built from another dataset's index (training and normalization statistics stay ICBHI)
        self.ood = dict(index_csv=ood_index_csv, data_dir=ood_data_dir, split=ood_split) if ood_index_csv else None
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
        """Original utils.get_mean_and_std -- averages per-image channel mean/std over the training set with augmentation enabled."""
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
        logger.info("RespireNet normalization statistics (train, with augmentation): mean %s std %s", mean.tolist(), std.tolist())
        return tuple(mean.tolist()), tuple(std.tolist())

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_ds is None:
            self.train_ds = RespireNetChunkDataset(train_flag=True, augment=self.augment, **self.kw)
            if self.mean is None or self.std is None:
                self.mean, self.std = self._compute_mean_std(self.train_ds)
            self.train_ds.set_normalization(self.mean, self.std)
        if self.test_ds is None and self.ood is not None:
            self.test_ds = RespireNetIndexChunkDataset(self.ood["index_csv"], self.ood["data_dir"], split=self.ood["split"],
                                                       sample_rate=self.kw["sample_rate"], desired_length=self.kw["desired_length"],
                                                       res_type=self.kw["res_type"], pad_seed=self.kw["pad_seed"])
            if self.mean is not None and self.std is not None:
                self.test_ds.set_normalization(self.mean, self.std)
        if self.test_ds is None:
            self.test_ds = RespireNetChunkDataset(train_flag=False, augment=False, **self.kw)
            if self.mean is None or self.std is None:  # evaluation only (test without fit): eval.py constants
                self.mean, self.std = DEFAULT_MEAN, DEFAULT_STD
            self.test_ds.set_normalization(self.mean, self.std)

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.weighted_sampler:
            # Original train.py:114-118 -- inverse class probability of chunk labels
            w = 1.0 / torch.tensor([self.train_ds.class_probs[l] for l in self.train_ds.labels], dtype=torch.float32)
            sampler = torch.utils.data.WeightedRandomSampler(w, len(self.train_ds))
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, shuffle=sampler is None,
                          num_workers=self.num_workers, collate_fn=respirenet_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=respirenet_collate)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
