"""
Prototype 논문(Ren et al., ICASSP 2022) 의 ICBHI 데이터 경로 — 공식 코드 충실 이식 (P0).

출처: https://github.com/L3S/PrototypeSound (MIT)
- `preprocessing/preprocessing.py`: 녹음을 **4 kHz 로 리샘플 → Butterworth 밴드패스 100–1800 Hz(order 5) → 사이클 분할**.
  즉 밴드패스는 **분할 전 녹음 전체**에 걸린다(분할 후 걸면 사이클 경계에 필터 과도응답이 남아 값이 달라진다).
- `baseline_cnn/utils/config.py`: cycle_len = 16000 샘플(= 4 s @4 kHz).
- `baseline_cnn/utils/data_generator.py:77-92`: 길면 **학습은 랜덤 크롭 / 평가는 센터 크롭**, 짧으면
  `np.pad(..., mode='wrap')` = 순환 반복 패딩. 정확히 같으면 그대로.

원본 그대로 유지하는 quirk:
- 공식은 사이클을 wav 로 미리 잘라 저장한 뒤 학습 때 다시 읽는다(리샘플·필터가 1회). 여기서는 원본 데이터 불변
  원칙에 따라 파일을 만들지 않고 같은 순서(로드→리샘플→밴드패스→분할)를 런타임에 재현한다. 결과 샘플은 동일하다.
- 툴킷 기본 `load_audio` 의 fade(sr/16) 는 공식에 없으므로 끈다(`fade=False`).
- 랜덤 크롭은 워커·epoch 마다 달라야 하므로 `torch.randint` 를 쓴다(공식은 `random.randint`). 분포는 같고
  구현상 시드 스트림만 다르다.
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
    """공식 preprocessing.butter_bandpass_filter — scipy butter(order) + lfilter (영위상 아님)."""

    def __init__(self, lowcut: float = 100.0, highcut: float = 1800.0, fs: int = 4000, order: int = 5):
        nyq = 0.5 * fs
        self.b, self.a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        y = lfilter(self.b, self.a, x, axis=-1)
        return torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32)


def fix_length(audio: torch.Tensor, cycle_len: int, train: bool) -> torch.Tensor:
    """data_generator.read_batch_audio — 길면 랜덤(train)/센터(eval) 크롭, 짧으면 wrap 패딩."""
    n = int(audio.shape[-1])
    if n == cycle_len:
        return audio
    if n > cycle_len:
        start = int(torch.randint(0, n - cycle_len + 1, (1,)).item()) if train else int((n - cycle_len) / 2)
        return audio[..., start:start + cycle_len]
    reps = -(-cycle_len // n)  # ceil — np.pad(mode='wrap') 과 동일한 순환 반복
    return audio.repeat(*([1] * (audio.dim() - 1)), reps)[..., :cycle_len]


class ProtoIndexDataset(IndexDataset):
    """IndexDataset + (녹음 전체 밴드패스 → 사이클 분할 → 4 s 크롭/패딩)."""

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
            raise ValueError(f"protonet 데이터 경로는 sample_level=cycle 만 지원한다 (got {level!r})")

        # 공식 순서: 로드+리샘플(4 kHz) → 밴드패스(녹음 전체) → 사이클 분할
        wav = load_audio(wav_path, target_sampling_rate=self.sampling_rate, fade=self.fade)
        if self.bandpass is not None:
            wav = self.bandpass(wav)
        t0, t1 = float(row["t0"]), float(row["t1"])
        audio = slice_audio(wav, t0=t0, t1=t1, sampling_rate=self.sampling_rate)
        audio = fix_length(audio, self.cycle_len, self.is_train)

        label = row["label"]
        return {
            "audio": audio,
            "label": None if (label != label) else int(label),  # NaN 체크
            "sample_id": str(row["sample_id"]),
            "recording_path": str(row["recording_path"]),
            "sample_level": level,
            "t0": t0,
            "t1": t1,
        }


class ProtoNetDataModule(IndexDataModule):
    """
    IndexDataModule + protonet 전용 데이터셋 주입 + balanced class weight 계산.

    class weight 는 공식 `data_generator.calculate_class_weight`(sklearn compute_class_weight 'balanced')
    와 동일하며, `train_class_weight` 속성으로 노출한다(콜백이 모델에 주입).
    """

    def __init__(self, *args, bandpass: bool = True, cycle_len: int = 16000, **kwargs):
        self.bandpass = bool(bandpass)
        self.cycle_len = int(cycle_len)
        self.train_class_weight: Optional[list] = None
        super().__init__(*args, **kwargs)
        # P2-8: 예전엔 super().setup() 이 만든 IndexDataset 을 `ProtoIndexDataset.__new__` + `__dict__.update` 로 바꿔치기했다.
        # 이제 생성 시점에 ProtoIndexDataset 을 쓰고 전용 인자는 `_index_dataset_kwargs` 훅으로 넘긴다
        # (config 의 index_dataset_cls 가 IndexDataset 이어도 동작 — 옛 config 무수정).
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
