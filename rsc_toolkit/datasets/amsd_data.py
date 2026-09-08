"""
AMS-D 데이터 파이프라인 — 공식 loader/mydataset.py ICBHIDataset.__getitem__ (:94-132) 의 fbank 이후 동작을 툴킷 IndexDataModule 위에 재현.

공식 순서 (train):  kaldi fbank(raw, 128 mel, 10 ms, hanning, dither 0) → 798 프레임 미만이면 **raw 영역 0-패딩**, 초과면 절단(:83-90, fallback)
                    → FrequencyMasking(freqm//2) → TimeMasking(timem//2) (torchaudio, 값 0, 각 1회; :105-117)
                    → 정규화 (x-mean)/(2·std) (:121) → [epoch ≥10 잡음+roll — 모델 training_forward 가 epoch 기준으로 적용]
        (eval):     fbank → (0-패딩) → 정규화.
툴킷 기본 MelBankAudioProcessor 는 정규화 **후** 0-패딩(패딩 값 0 ≠ 공식의 (0-mean)/(2std)=+0.467) 이므로, 특징 캐시는 normalize=false 로
raw fbank 를 저장하고 이 모듈이 마스킹 → 정규화를 샘플별로 수행한다. 툴킷 SpecAugment(icbhi_ast_sup: 48/160 ×2, mean) 와 정책이
달라 쓰지 않는다 (data config specaugment.enabled=false).

클립 길이(파형 단계) — **가정(저자 질문 #10)**: 공식 코드에는 클립 생성 스크립트가 없고 논문 §3.1 은 "16 kHz 리샘플 + 8 s loop-pad" 를
명시한다. target_length 798 = (8·16000−400)/160+1 은 정확히 8 s 클립의 프레임 수라 저자 데이터에서는 fbank 0-패딩 분기가 실행되지 않는다.
따라서 기본값은 논문 명시(desired_length_sec 8.0, pad_type repeat = 반복 패딩, fade 없음) 이고, 파형 패딩 없이 fbank 영역 0-패딩으로
돌리는 경로(data.params.desired_length_sec=null data.params.pad_type=zero) 는 두 번째 가정으로 남겨 둔다. 두 경로 모두 저자 확인 전.

특징 캐시: FeatureCache 키는 MelBankAudioProcessor 설정만 포함하고 파형 패딩(desired_length_sec/pad_type)·fade 는 포함하지 않는다
(feature_cache.py:_create_cache_key). 패딩 가정을 바꿔도 캐시가 조용히 재사용되는 사고를 막기 위해 이 모듈은 cache_dir 아래에
`pad-<pad_type>-<desired_length_sec>-fade<fade>` 하위 디렉터리를 붙인다.

미상 값(누락 config_train.yaml, 질문 #3): freqm/timem — 논문 §3.1 은 SpecAugment 를 언급하지 않아(잡음·time-rolling 만) 0/0(미적용) 도 동등한
후보이나 **가정** 으로 AST AudioSet 관례 freqm 48 / timem 192 를 넣는다(코드가 //2 하므로 실효 24/96 — 관례의 실효값 48/192 와 다름).
mean/std 도 논문 미기재 → AST AudioSet 상수(−4.2677393 / 4.5689974) 가정. 값은 data config 로 바꾼다.

샘플 인덱스: train 배치의 4번째 원소(meta_labels) 로 IndexDataset 행 번호(공식 train 인덱스 순서 고정 = index_cycle.csv 의 공식 split
필터 순서) 를 넘긴다 → 모델의 per-sample 버퍼(all_predictions/ids_masked)·잡음 시드 2345+epoch+index 가 이 번호에 묶인다.
원본의 dataset_aug 로컬 index 충돌 quirk(원본/증강본 i 번째가 같은 슬롯 공유) 는 dataset_aug 없이 학습하므로 발생하지 않는다.
val(=공식 test) 배치는 3-튜플(meta 없음) → PLAdapter 가 meta 를 넘기지 않아 모델은 AST 경로만 탄다.

미재현 공식 RNG quirk(문서화만): (a) epoch ≥10 마다 DataLoader 를 generator seed 2345 로 재생성(myloader.py:301, main.py:213-214) →
epoch ≥10 의 배치 순서가 매 epoch 동일하고 drop_last 로 항상 같은 샘플이 탈락; (b) mydataset.py:127 torch.manual_seed(2345+epoch+index) 가
워커 전역 RNG 를 재시드해 같은 워커의 **다음** 샘플의 Freq/TimeMasking 마스크까지 (epoch, 직전 index) 의 결정적 함수가 됨.
툴킷은 Lightning seed 기반 독립 스트림(잡음·roll 은 모델 안에서 (seed, epoch, index) 파생 생성기).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule


class AMSDFeatureDataset(Dataset):
    """IndexFeatureDataset 래퍼: (raw fbank [F,T]) → [train: freq/time 마스킹] → 정규화 → (feats, label, sample_id[, index])."""

    def __init__(self, inner: IndexFeatureDataset, training: bool, freqm: int, timem: int, mean: float, std: float,
                 return_index: bool):
        import torchaudio

        self.inner = inner
        self.training = bool(training)
        self.freqm, self.timem = int(freqm), int(timem)
        self.mean, self.std = float(mean), float(std)
        self.return_index = bool(return_index)
        # mydataset.py:110,113 — 파라미터 freqm//2, timem//2 (0 이면 건너뜀)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(self.freqm // 2) if self.freqm != 0 else None
        self.time_mask = torchaudio.transforms.TimeMasking(self.timem // 2) if self.timem != 0 else None

    def __len__(self):
        return len(self.inner)

    def transform(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (F, T) raw fbank (798 프레임 0-패딩/절단 완료). 공식과 같은 순서로 마스킹 → 정규화."""
        x = feats
        if self.training:
            x = x.unsqueeze(0)  # (1, F, T) — 원본은 (T,F)→transpose→(1,F,T) 로 같은 축 배치
            if self.freq_mask is not None:
                x = self.freq_mask(x)
            if self.time_mask is not None:
                x = self.time_mask(x)
            x = x.squeeze(0)
        return (x - self.mean) / (self.std * 2)

    def __getitem__(self, i: int):
        out = self.inner[i]
        feats = self.transform(out[0])
        if self.return_index:
            return feats, out[1], out[2], int(i)
        return feats, out[1], out[2]


def _splits_of(ds) -> List[str]:
    """IndexDataset(.split) 또는 ConcatDataset(.datasets) 의 split 이름 목록."""
    if hasattr(ds, "datasets"):
        return [str(getattr(d, "split")) for d in ds.datasets]
    return [str(getattr(ds, "split"))]


class AMSDDataModule(IndexDataModule):
    meta_kind = "amsd_sample_index"  # 학습 batch[2] = 샘플 인덱스 (amsd_return_index) — ast_amsd 전용
    """IndexDataModule + AMS-D 샘플별 마스킹/정규화 래퍼 + train 배치 샘플 인덱스(meta) 부착 + 패딩 설정별 캐시 분리."""

    def __init__(self, *args, amsd_freqm: int = 48, amsd_timem: int = 192, amsd_mean: float = -4.2677393,
                 amsd_std: float = 4.5689974, amsd_return_index: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.features_cfg.normalize:
            raise ValueError("AMSDDataModule 은 features.normalize=false (raw fbank 캐시) 여야 한다 — 정규화는 마스킹 뒤에 한다")
        if self.specaugment_cfg.enabled:
            raise ValueError("AMSDDataModule 은 툴킷 SpecAugment 를 쓰지 않는다 (specaugment.enabled=false)")
        if self.meta_label_mode is not None:
            raise ValueError("AMSDDataModule 은 meta 자리에 샘플 인덱스를 넣는다 (meta_label_mode 는 None 이어야 함)")
        if list(self.train_splits) == list(self.val_splits):
            raise ValueError("AMSDDataModule 은 split 이름으로 train/val 을 구분한다 — train_splits 와 val_splits 가 같을 수 없다")
        if not self.features_cfg.cache_dir:
            raise ValueError("AMSDDataModule 은 features.cache_dir 이 필요하다 (패딩 설정별 하위 디렉터리를 붙인다)")
        self.amsd_freqm, self.amsd_timem = int(amsd_freqm), int(amsd_timem)
        self.amsd_mean, self.amsd_std = float(amsd_mean), float(amsd_std)
        self.amsd_return_index = bool(amsd_return_index)
        # 캐시 키(processor config) 에 없는 파형 전처리 설정을 디렉터리 이름으로 분리 — 패딩 가정을 바꿔도 기존 캐시가 재사용되지 않게
        self.pad_cache_subdir = f"pad-{self.pad_type}-{self.desired_length_sec}-fade{self.fade}"
        self.features_cfg.cache_dir = str(Path(self.features_cfg.cache_dir) / self.pad_cache_subdir)

    def _is_train_base(self, base_ds) -> bool:
        return _splits_of(base_ds) == [str(s) for s in self.train_splits]

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None):
        inner = super()._wrap_with_features(base_ds, feature_transform=None, meta_label_fn=None)
        training = self._is_train_base(base_ds)  # split 이름으로 명시 판정 (setup() 호출 순서에 의존하지 않음)
        return AMSDFeatureDataset(inner, training=training, freqm=self.amsd_freqm, timem=self.amsd_timem,
                                  mean=self.amsd_mean, std=self.amsd_std, return_index=training and self.amsd_return_index)

    def _warmup_feature_cache(self, ds, *, split_name: str) -> None:  # 래퍼 안쪽 IndexFeatureDataset 으로 캐시 워밍
        inner = getattr(ds, "inner", ds)
        super()._warmup_feature_cache(inner, split_name=split_name)

    @property
    def num_train_samples(self) -> Optional[int]:
        return None if self.train_ds is None else len(self.train_ds)
