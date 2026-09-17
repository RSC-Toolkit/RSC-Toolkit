# Portions of this file are adapted from BTS-CARD (https://github.com/RSC-Toolkit/BTS-CARD); that code is under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Data path for **training BTS-family models (BTS-CARD, BTS) directly on SPRSound** (SPR->SPR).

Why a separate module -- three things do not work with the existing path:

1. **meta must be present in val batches too.** IndexDataModule attaches `meta_label_fn` to train only
   (`index_datamodule.py:519-526` -- val is wrapped as `_wrap_with_features(..., feature_transform=None)` without meta).
   BTS-family models also use meta in `validation_step`, so validation fails without it
   (`ValueError` asking for the 7-field meta in the batch).
2. **meta is a 7-vector, not a single integer.** `IndexFeatureDataset` casts with `int(meta_label_fn(item))` and
   `_tuple_collate` stacks with `dtype=torch.long`. Neither passes a float 7-vector through.
3. **SPRSound lacks 3 of the 7 fields.** adult_BMI, child_weight and child_height are not recorded at all.
   They are filled with **-1**, the original code's missing-value marker. `meta_fields_from_meta7` does not read these three slots
   (it reads only age=0, sex=1, loc=5, dev=6), so they enter neither the sentence nor the discriminator labels -- i.e. **no values are invented**.

The label space is the original 7 classes (the official definition used by papers that take SPRSound as the main task).
Do not confuse this with the 7->4 merge of the OOD view (`sprsound_ood.py`).
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

# Value the original code uses for missing fields (same convention as icbhi/meta_description.py meta7_from_index_meta)
MISSING = -1.0


def meta7_from_sprsound_meta(meta: Dict[str, Any]) -> List[float]:
    """
    SPRSound index meta dict -> the 7-field vector expected by BTS-family models.

    The ICBHI 7-field layout is [age, sex, adult_BMI, child_weight, child_height, loc_id, dev_id] and
    `bts_card.training_forward` uses `meta7[:,5]` (loc) and `meta7[:,6]` (dev) as discriminator labels.
    SPRSound has only age, gender, location_id and device_id, so the other three are -1 (missing).

    Note: loc_id is SPRSound's p1~p4 (0~3), which **means something different** from ICBHI's 7 chest locations.
    Sentence generation must therefore use the SPRSound dialect (`meta_dialect="sprsound"`) --
    mapping to the ICBHI vocabulary would invent auscultation locations that do not exist.
    """
    loc = meta.get("location_id")
    if loc is None:
        loc = LOC_CODE_TO_ID[str(meta["recording_location"])]
    return [
        float(meta["age"]),
        float(int(float(meta["gender"]))),
        MISSING,   # adult_BMI      -- not in SPRSound
        MISSING,   # child_weight   -- not in SPRSound
        MISSING,   # child_height   -- not in SPRSound
        float(int(loc)),
        float(int(float(meta.get("device_id", 0)))),
    ]


class _BTSCARDSprsFeatureDataset(IndexFeatureDataset):
    """(feats, label, sample_id, meta7 list[float]) -- yields a 7-vector instead of an integer meta."""

    def __init__(self, index_ds, feature_processor, feature_transform=None):
        super().__init__(index_ds, feature_processor, feature_transform=feature_transform, meta_label_fn=None)

    def __getitem__(self, i):
        item = self.index_ds[i]
        audio = item["audio"].squeeze()
        feats = self.fp.make_features_cached(audio, sample_id=item["sample_id"])
        if self.feature_transform is not None:
            feats = self.feature_transform(feats)
        return feats, item["label"], item["sample_id"], meta7_from_sprsound_meta(item["meta"])


from rsc_toolkit.datasets import collate as _collate  # single collate implementation

# Stack meta as a (B, 7) float32 tensor instead of tuple_collate, which casts meta to long
btscard_sprs_collate = _collate.btscard_sprs_collate


class BTSCARDSprsDataModule(IndexDataModule):
    meta_kind = "bts_meta7"  # batch[2] = meta7 float vector (both train and val) -- bts/bts_card/audioclap
    """IndexDataModule + meta7 attached to **both train and val** batches (for BTS-family SPR->SPR training)."""

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
        # meta_label_fn is ignored -- this path always attaches meta7 (val included).
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
            raise RuntimeError("train_ds is None. Call setup('fit') first.")
        return self._loader(self.train_ds, self.shuffle_train, self.drop_last_train)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds is None. Call setup('fit') first.")
        return self._loader(self.val_ds, False, False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("test_ds is None. Call setup('test') first.")
        return self._loader(self.test_ds, False, False)
