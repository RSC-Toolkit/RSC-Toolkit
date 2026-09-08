"""
LoRA-RSC(ICASSP 2025) 파형 잡음의 **대안 해석**(1D 파형 → 샘플별 N(0, 0.3²) 가우시안 잡음) — 플래그용 IndexDataset 서브클래스.

공개 코드 dataset/icbhi.py:241-245 는 (1,N) 2D 배열에 `np.random.randn(len(audio_numpy))` 를 더하므로 len=1 → 클립당 **스칼라 DC 오프셋**
이고, 이는 kaldi fbank 의 remove_dc_offset 으로 제거되어 특징이 바뀌지 않는다(골든 features.pt, max|diff| 6.4e-5). 그러나 같은 골든의
orig_2d_call 이 보여주듯 공개 코드는 이 (1,N) 호출에서 4.28.1 ASTFeatureExtractor → kaldi AssertionError 로 **실행 자체가 불가**하므로
저자의 실제 실행은 이 지점에서 공개 코드와 달랐다. 가장 자연스러운 수정(audio.numpy()[0] 등 1D 화)이었다면 len=128000 → 샘플별
N(0,0.3²) 잡음이 train/test 캐시에 1회(고정) 들어가는 전혀 다른 파이프라인이 된다. 어느 쪽인지 코드로 확정 불가(저자 질문 #6,
docs/AUTHOR_QUESTIONS_2026-08-26.md) → 기본 config(icbhi_lorarsc) 는 잡음 없음(스칼라 해석 = 무효), 이 클래스 + data=icbhi_lorarsc_noise
가 1D 해석이다.

재현 방식: 원본은 np.random.seed(10) 상태에서 데이터셋 순서대로 1회 뽑아 캐시했으므로(전역 RNG 순서 의존, 정확 재현 불가) 여기서는
sample_id 와 noise_seed 로 결정된 per-sample RNG 를 써서 **캐시 1회·train/test 공통·결정적** 성질만 재현한다.
float64 로 더한 뒤 float32 로 캐스트하는 순서는 원본(astype(float64) → 추출기 내부 float32) 과 같다.
특징 캐시 키는 processor config 만으로 만들어지므로 잡음 버전은 반드시 **다른 cache_dir** 를 써야 한다(icbhi_lorarsc_noise.yaml 참고).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np
import torch

from .index_dataset import IndexDataset


class LoRARSCNoisyIndexDataset(IndexDataset):
    """IndexDataset + 샘플별 가우시안 파형 잡음(std=noise_std, 결정적).

    생성자 인자 `noise_std`/`noise_seed` 로 조절한다 — config 에서는 `data.params.index_dataset_params: {noise_std: 0.3, noise_seed: 10}`
    (P2-8; 예전엔 IndexDataModule 이 커스텀 클래스에 인자를 못 넘겨 클래스 속성으로만 바꿀 수 있었다). 클래스 속성은 기본값이다.
    """

    noise_std: float = 0.3      # dataset/icbhi.py:243 `0.3 * noise`
    noise_seed: int = 10        # 원본 --seed 기본값 10 (np.random.seed(10)); per-sample 파생 시드의 기저

    def __init__(self, *args, noise_std: float | None = None, noise_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if noise_std is not None:
            self.noise_std = float(noise_std)
        if noise_seed is not None:
            self.noise_seed = int(noise_seed)

    @classmethod
    def sample_rng(cls, sample_id: str, noise_seed: int | None = None) -> np.random.Generator:
        seed = cls.noise_seed if noise_seed is None else int(noise_seed)
        h = hashlib.md5(f"{seed}:{sample_id}".encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    @classmethod
    def add_noise(cls, audio: torch.Tensor, sample_id: str, noise_std: float | None = None,
                  noise_seed: int | None = None) -> torch.Tensor:
        std = cls.noise_std if noise_std is None else float(noise_std)
        if std <= 0:
            return audio
        x = audio.detach().cpu().numpy().astype(np.float64)
        noise = cls.sample_rng(sample_id, noise_seed).standard_normal(size=x.shape)  # 1D 해석: 샘플마다 독립 N(0,1)
        return torch.from_numpy((x + std * noise).astype(np.float32))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        out["audio"] = self.add_noise(out["audio"], out["sample_id"], self.noise_std, self.noise_seed)
        return out
