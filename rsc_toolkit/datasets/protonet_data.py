"""
ICBHI data path of the Prototype paper (Ren et al., ICASSP 2022) -- exact port of the official code (verified against goldens).

Source: https://github.com/L3S/PrototypeSound (MIT)
- `preprocessing/preprocessing.py`: recordings are **resampled to 4 kHz -> Butterworth band-pass 100-1800 Hz (order 5) -> split into cycles**.
  That is, the band-pass is applied to **the whole recording before splitting** (filtering after splitting leaves filter transients at cycle boundaries and changes the values).
- `baseline_cnn/utils/config.py`: cycle_len = 16000 samples (= 4 s @4 kHz).
- `baseline_cnn/utils/data_generator.py:77-92`: if longer, **random crop for training / center crop for evaluation**; if shorter,
  `np.pad(..., mode='wrap')` = circular repeat padding. Unchanged if exactly equal.

Quirks kept as in the original code:
- The official code pre-cuts cycles into wav files and reads them back during training (resampling and filtering happen once). Here, to keep raw data
  immutable, no files are created and the same order (load -> resample -> band-pass -> split) is reproduced at runtime. The resulting samples are identical.
- The fade (sr/16) of the toolkit's default `load_audio` is not in the official code, so it is disabled (`fade=False`).
- Random crops must differ per worker and epoch, so `torch.randint` is used (the official code uses `random.randint`). The distribution is the same;
  only the seed stream differs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.signal import butter, lfilter

from rsc_toolkit.datasets.icbhi.processing import load_audio, slice_audio
from rsc_toolkit.datasets.index_dataset import IndexDataset
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

logger = logging.getLogger(__name__)


class ButterBandpass:
    """Official preprocessing.butter_bandpass_filter -- scipy butter(order) + lfilter (not zero-phase)."""

    def __init__(self, lowcut: float = 100.0, highcut: float = 1800.0, fs: int = 4000, order: int = 5):
        nyq = 0.5 * fs
        self.b, self.a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        y = lfilter(self.b, self.a, x, axis=-1)
        return torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32)


def fix_length(audio: torch.Tensor, cycle_len: int, train: bool) -> torch.Tensor:
    """data_generator.read_batch_audio -- random (train) / center (eval) crop if longer, wrap padding if shorter."""
    n = int(audio.shape[-1])
    if n == cycle_len:
        return audio
    if n > cycle_len:
        start = int(torch.randint(0, n - cycle_len + 1, (1,)).item()) if train else int((n - cycle_len) / 2)
        return audio[..., start:start + cycle_len]
    reps = -(-cycle_len // n)  # ceil -- same circular repetition as np.pad(mode='wrap')
    return audio.repeat(*([1] * (audio.dim() - 1)), reps)[..., :cycle_len]


class ProtoIndexDataset(IndexDataset):
    """IndexDataset + (band-pass over the whole recording -> cycle split -> 4 s crop/pad)."""

    def __init__(self, *args, bandpass: bool = True, cycle_len: int = 16000, is_train: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_len = int(cycle_len)
        self.is_train = bool(is_train)
        self.bandpass = ButterBandpass(fs=self.sampling_rate) if bandpass else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        wav_path = self._resolve_recording_path(str(row["recording_path"]))
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        level = str(row["sample_level"])
        if level != "cycle":
            raise ValueError(f"the protonet data path supports only sample_level=cycle (got {level!r})")

        # Official order: load + resample (4 kHz) -> band-pass (whole recording) -> cycle split
        wav = load_audio(wav_path, target_sampling_rate=self.sampling_rate, fade=self.fade)
        if self.bandpass is not None:
            wav = self.bandpass(wav)
        t0, t1 = float(row["t0"]), float(row["t1"])
        audio = slice_audio(wav, t0=t0, t1=t1, sampling_rate=self.sampling_rate)
        audio = fix_length(audio, self.cycle_len, self.is_train)

        label = row["label"]
        return {
            "audio": audio,
            "label": None if (label != label) else int(label),  # NaN check
            "sample_id": str(row["sample_id"]),
            "recording_path": str(row["recording_path"]),
            "sample_level": level,
            "t0": t0,
            "t1": t1,
        }


class ProtoNetDataModule(IndexDataModule):
    """
    IndexDataModule + protonet-specific dataset injection + balanced class weight computation.

    The class weight is identical to the official `data_generator.calculate_class_weight` (sklearn compute_class_weight 'balanced')
    and is exposed as the `train_class_weight` attribute (a callback injects it into the model).
    """

    def __init__(self, *args, bandpass: bool = True, cycle_len: int = 16000, **kwargs):
        self.bandpass = bool(bandpass)
        self.cycle_len = int(cycle_len)
        self.train_class_weight: Optional[list] = None
        super().__init__(*args, **kwargs)
        # ProtoIndexDataset is used at construction time; its specific arguments go through the `_index_dataset_kwargs` hook
        # (works even if the config's index_dataset_cls is IndexDataset -- older configs need no changes).
        if not (isinstance(self.index_dataset_cls, type) and issubclass(self.index_dataset_cls, ProtoIndexDataset)):
            self.index_dataset_cls = ProtoIndexDataset

    def _index_dataset_kwargs(self, *, split: str, role: Optional[str] = None):
        kwargs = super()._index_dataset_kwargs(split=split, role=role)
        kwargs.update(cycle_len=self.cycle_len, bandpass=self.bandpass, is_train=(role == "train"))
        return kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self._compute_class_weight()

    def _compute_class_weight(self) -> None:
        ds = getattr(self, "train_ds", None)
        base = getattr(ds, "index_ds", ds)
        if base is None:
            return
        from sklearn.utils.class_weight import compute_class_weight

        y = base.df["label"].dropna().astype(int).to_numpy()
        classes = np.sort(np.unique(y))
        self.train_class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y).tolist()
        logger.info("protonet balanced class weight: %s", [round(w, 4) for w in self.train_class_weight])
