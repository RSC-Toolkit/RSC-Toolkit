"""
BTS 계열(BTS-CARD·BTS)을 **SPRSound 로 직접 학습**할 때 쓰는 데이터 경로 (SPR→SPR).

왜 별도 모듈인가 — 세 가지가 기존 경로로는 안 된다:

1. **meta 가 val 배치에도 있어야 한다.** IndexDataModule 은 `meta_label_fn` 을 train 에만 붙인다
   (`index_datamodule.py:519-526` — val 은 `_wrap_with_features(..., feature_transform=None)` 로 meta 없이 감싼다).
   BTS 계열은 `validation_step` 에서도 meta 를 쓰므로 그대로 두면 검증에서 죽는다 —
   실제로 `sprs-bts-s1/s2` 가 `ValueError: BTS 는 배치에 meta(7필드) 가 필요하다` 로 EXIT=1 났다.
2. **meta 가 정수 하나가 아니라 7-벡터다.** `IndexFeatureDataset` 은 `int(meta_label_fn(item))` 로 정수화하고
   `_tuple_collate` 도 `dtype=torch.long` 으로 쌓는다. 둘 다 float 7-벡터를 통과시키지 못한다.
3. **SPRSound 에는 7필드 중 3개가 없다.** adult_BMI·child_weight·child_height 는 기록 자체가 없다.
   원본의 결측 표기 그대로 **-1** 을 넣는다. 이 세 자리는 `meta_fields_from_meta7` 이 읽지 않으므로
   (age=0, sex=1, loc=5, dev=6 만 읽는다) 문장에도 판별기 라벨에도 들어가지 않는다 — 즉 **값을 지어내지 않는다**.

라벨 공간은 7-class 원본 그대로다(SPRSound 를 주 태스크로 쓴 논문들의 공식 정의).
OOD 뷰(`sprsound_ood.py`)의 7→4 병합과 혼동하지 말 것.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset
from rsc_toolkit.datasets.sprsound_ood import LOC_CODE_TO_ID
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

logger = logging.getLogger(__name__)

# 원본이 결측에 쓰는 값 (icbhi/meta_description.py meta7_from_index_meta 와 동일 규약)
MISSING = -1.0


def meta7_from_sprsound_meta(meta: Dict[str, Any]) -> List[float]:
    """
    SPRSound 인덱스 meta dict → BTS 계열이 기대하는 7필드 벡터.

    ICBHI 7필드 배치는 [age, sex, adult_BMI, child_weight, child_height, loc_id, dev_id] 이고
    `bts_card.training_forward` 가 판별기 라벨로 `meta7[:,5]`(loc)·`meta7[:,6]`(dev) 를 쓴다.
    SPRSound 는 age·gender·location_id·device_id 넷만 가지므로 나머지 셋은 -1(결측)이다.

    주의: loc_id 는 SPRSound 의 p1~p4(0~3)이고 ICBHI 의 흉부 위치 7종과 **의미가 다르다**.
    따라서 문장 생성은 반드시 SPRSound 방언(`meta_dialect="sprsound"`)을 써야 한다 —
    ICBHI 어휘로 옮기면 없는 청진 위치를 지어내게 된다.
    """
    loc = meta.get("location_id")
    if loc is None:
        loc = LOC_CODE_TO_ID[str(meta["recording_location"])]
    return [
        float(meta["age"]),
        float(int(float(meta["gender"]))),
        MISSING,   # adult_BMI      — SPRSound 에 없음
        MISSING,   # child_weight   — SPRSound 에 없음
        MISSING,   # child_height   — SPRSound 에 없음
        float(int(loc)),
        float(int(float(meta.get("device_id", 0)))),
    ]


class _BTSCARDSprsFeatureDataset(IndexFeatureDataset):
    """(feats, label, sample_id, meta7 list[float]) — 정수 meta 대신 7-벡터를 낸다."""

    def __init__(self, index_ds, feature_processor, feature_transform=None):
        super().__init__(index_ds, feature_processor, feature_transform=feature_transform, meta_label_fn=None)

    def __getitem__(self, i):
        item = self.index_ds[i]
        audio = item["audio"].squeeze()
        feats = self.fp.make_features_cached(audio, sample_id=item["sample_id"])
        if self.feature_transform is not None:
            feats = self.feature_transform(feats)
        return feats, item["label"], item["sample_id"], meta7_from_sprsound_meta(item["meta"])


from rsc_toolkit.datasets import collate as _collate  # P2-6: collate 단일 구현

# meta 를 long 으로 접는 tuple_collate 대신 (B, 7) float32 텐서로 쌓는다
btscard_sprs_collate = _collate.btscard_sprs_collate


class BTSCARDSprsDataModule(IndexDataModule):
    meta_kind = "bts_meta7"  # batch[2] = meta7 float 벡터 (train·val 양쪽) — bts/bts_card/audioclap
    """IndexDataModule + meta7 를 **train·val 양쪽** 배치에 붙인다 (BTS 계열 SPR→SPR 학습용)."""

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None, split_role=None):
        import pathlib

        from rsc_toolkit.datasets.feature_cache import FeatureCache

        cache_dir = (
            pathlib.Path(self.features_cfg.cache_dir)
            if self.features_cfg.cache_dir
            else (self.outputs_directory / "features_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        feature_cache = FeatureCache(cache_dir=cache_dir)
        fp = self._build_feature_processor(feature_cache)
        feature_cache.configure(
            fp.get_processor_config(),
            processor_name=fp.__class__.__name__,
            audio_chain=self._audio_chain(),
        )
        # meta_label_fn 은 무시한다 — 이 경로는 항상 meta7 을 붙인다(val 포함).
        return _BTSCARDSprsFeatureDataset(base_ds, fp, feature_transform=feature_transform)

    def _loader(self, ds, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            collate_fn=btscard_sprs_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds is None. setup('fit') 를 먼저 불러야 한다.")
        return self._loader(self.train_ds, self.shuffle_train, self.drop_last_train)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds is None. setup('fit') 를 먼저 불러야 한다.")
        return self._loader(self.val_ds, False, False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("test_ds is None. setup('test') 를 먼저 불러야 한다.")
        return self._loader(self.test_ds, False, False)
