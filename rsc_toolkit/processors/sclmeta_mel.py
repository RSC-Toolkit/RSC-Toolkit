"""
SCL-metadata(Moummad & Farrugia, ICASSP 2023 / arXiv 2210.16192)의 스펙트로그램 — 공식 코드 충실 이식.

출처: https://github.com/ilyassmoummad/scl_icbhi2017 main.py:29-34,59-62 + utils.py.
샘플별 단계(캐시 가능): torchaudio MelSpectrogram(sample_rate 16000(기본값), n_fft 1024, win 1024, hop 512,
n_mels 64, f_min 50, f_max 2000, power 2) → AmplitudeToDB(power, top_db None) → (1, 64, 251) for 8초.
배치 단계(모델 안, 배치 의존): 배치 전체 min-max → [SpecAugment] → Standardize(0.3690, 0.0255).
"""
from typing import Any, Dict, Optional

import torch
import torchaudio.transforms as T

from rsc_toolkit.processors.base import BaseAudioProcessor

STANDARDIZE_MEAN = 0.3690  # utils.py Standardize 기본값 ("official split mean & std")
STANDARDIZE_STD = 0.0255


def batch_minmax(x: torch.Tensor) -> torch.Tensor:
    """utils.py Normalize.forward 의 뒷부분: 배치 텐서 전체의 min/max 로 [0,1]."""
    return (x - x.min()) / (x.max() - x.min())


def standardize(x: torch.Tensor, mean: float = STANDARDIZE_MEAN, std: float = STANDARDIZE_STD) -> torch.Tensor:
    return (x - mean) / std


class SCLMetaMelProcessor(BaseAudioProcessor):
    def __init__(self, sampling_rate: int = 16000, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512,
                 n_mels: int = 64, f_min: float = 50, f_max: float = 2000, **kwargs):
        self.sampling_rate = int(sampling_rate)
        self.cfg = dict(n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max)
        # 원본은 sample_rate 를 넘기지 않아 torchaudio 기본 16000 이 쓰인다 — 같은 값을 명시
        self.melspec = T.MelSpectrogram(sample_rate=self.sampling_rate, **self.cfg)
        self.power_to_db = T.AmplitudeToDB()
        super().__init__(**kwargs)

    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):
        # 예전 IndexDataModule._build_feature_processor 의 kind=="sclmeta_mel" 분기와 같은 생성 인자
        return cls(
            sampling_rate=sampling_rate, n_mels=opts["n_mels"], n_fft=opts["n_fft"],
            win_length=opts["win_length"] or opts["n_fft"], hop_length=opts["hop_length"],
            f_min=opts["f_min"], f_max=opts["f_max"], feature_cache=feature_cache,
        )

    def get_processor_config(self) -> Dict[str, Any]:
        return {"kind": "sclmeta_mel", "sampling_rate": self.sampling_rate, **self.cfg}

    def make_features(self, audio: torch.Tensor, sample_id: Optional[str] = None) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        with torch.no_grad():
            return self.power_to_db(self.melspec(audio.float()))  # (1, n_mels, frames)
