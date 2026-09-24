# Portions of this file are adapted from AMS-D (https://github.com/CcnNnn/AMS-D); that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
AMS-D data pipeline -- reproduces the post-fbank behaviour of the official loader/mydataset.py ICBHIDataset.__getitem__ (:94-132) on top of the toolkit IndexDataModule.

Official order (train): kaldi fbank(raw, 128 mel, 10 ms, hanning, dither 0) -> **zero-pad in the raw domain** if shorter than 798 frames, crop if longer (:83-90, fallback)
                    -> FrequencyMasking(freqm//2) -> TimeMasking(timem//2) (torchaudio, value 0, once each; :105-117)
                    -> normalize (x-mean)/(2*std) (:121) -> [epoch >= 10 noise + roll -- applied per epoch by the model's training_forward]
        (eval):     fbank -> (zero-pad) -> normalize.
The toolkit's default MelBankAudioProcessor zero-pads **after** normalization (pad value 0 != official (0-mean)/(2std)=+0.467), so the feature cache
stores raw fbank with normalize=false and this module applies masking -> normalization per sample. The toolkit SpecAugment (icbhi_ast_sup: 48/160 x2, mean)
follows a different policy and is not used (data config specaugment.enabled=false).

Clip length (waveform stage) -- **Assumption**: the official code has no clip-generation script, and the paper (Sec. 3.1) states "16 kHz resampling + 8 s loop-pad".
target_length 798 = (8*16000-400)/160+1 is exactly the frame count of an 8 s clip, so the fbank zero-padding branch never runs on the authors' data.
The default therefore follows the paper (desired_length_sec 8.0, pad_type repeat = repeat padding, no fade); the path without waveform padding and with
zero-padding in the fbank domain (data.params.desired_length_sec=null data.params.pad_type=zero) is kept as a second assumption. Neither is confirmed by the authors.

Feature cache: the FeatureCache key includes only the MelBankAudioProcessor config, not waveform padding (desired_length_sec/pad_type) or fade
(feature_cache.py:_create_cache_key). To keep a cache from being silently reused after the padding assumption changes, this module appends a
`pad-<pad_type>-<desired_length_sec>-fade<fade>` subdirectory under cache_dir.

Unknown values (config_train.yaml is missing): freqm/timem -- the paper (Sec. 3.1) does not mention SpecAugment (only noise and time-rolling), so 0/0 (disabled) is an
equally valid candidate, but the AST AudioSet convention freqm 48 / timem 192 is **assumed** (the code halves them, so the effective values are 24/96 -- unlike the convention's effective 48/192).
mean/std are not given in the paper either -> AST AudioSet constants (-4.2677393 / 4.5689974) assumed. Both are set in the data config.

Sample index: the 4th element of a train batch (meta_labels) carries the IndexDataset row number (fixed official train index order = the official split
filter order of index_cycle.csv) -> the model's per-sample buffers (all_predictions/ids_masked) and the noise seed 2345+epoch+index are tied to this number.
The original code's dataset_aug local-index collision quirk (the i-th original and augmented samples share a slot) does not occur because training runs without dataset_aug.
val (= official test) batches are 3-tuples (no meta) -> PLAdapter passes no meta and the model takes the AST path only.

Official RNG quirks not reproduced (documented only): (a) from epoch >= 10 the DataLoader is recreated with generator seed 2345 each epoch (myloader.py:301, main.py:213-214) ->
the batch order for epoch >= 10 is identical every epoch and drop_last always drops the same samples; (b) mydataset.py:127 torch.manual_seed(2345+epoch+index)
reseeds the worker-global RNG, so the Freq/TimeMasking masks of the **next** sample in the same worker become a deterministic function of (epoch, previous index).
The toolkit uses independent streams from the Lightning seed (noise and roll use generators derived from (seed, epoch, index) inside the model).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from rsc_toolkit.datasets.index_dataset import IndexFeatureDataset
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule


class AMSDFeatureDataset(Dataset):
    """IndexFeatureDataset wrapper: (raw fbank [F,T]) -> [train: freq/time masking] -> normalization -> (feats, label, sample_id[, index])."""

    def __init__(self, inner: IndexFeatureDataset, training: bool, freqm: int, timem: int, mean: float, std: float,
                 return_index: bool):
        import torchaudio

        self.inner = inner
        self.training = bool(training)
        self.freqm, self.timem = int(freqm), int(timem)
        self.mean, self.std = float(mean), float(std)
        self.return_index = bool(return_index)
        # mydataset.py:110,113 -- parameters freqm//2, timem//2 (skipped when 0)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(self.freqm // 2) if self.freqm != 0 else None
        self.time_mask = torchaudio.transforms.TimeMasking(self.timem // 2) if self.timem != 0 else None

    def __len__(self):
        return len(self.inner)

    def transform(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (F, T) raw fbank (already zero-padded/cropped to 798 frames). Masking -> normalization in the official order."""
        x = feats
        if self.training:
            x = x.unsqueeze(0)  # (1, F, T) -- the original goes (T,F)->transpose->(1,F,T), same axis layout
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
    """List of split names of an IndexDataset (.split) or ConcatDataset (.datasets)."""
    if hasattr(ds, "datasets"):
        return [str(getattr(d, "split")) for d in ds.datasets]
    return [str(getattr(ds, "split"))]


class AMSDDataModule(IndexDataModule):
    meta_kind = "amsd_sample_index"  # train batch[2] = sample index (amsd_return_index) -- ast_amsd only
    """IndexDataModule + AMS-D per-sample masking/normalization wrapper + train-batch sample index (meta) + cache separated per padding setting."""

    def __init__(self, *args, amsd_freqm: int = 48, amsd_timem: int = 192, amsd_mean: float = -4.2677393,
                 amsd_std: float = 4.5689974, amsd_return_index: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.features_cfg.normalize:
            raise ValueError("AMSDDataModule requires features.normalize=false (raw fbank cache) -- normalization happens after masking")
        if self.specaugment_cfg.enabled:
            raise ValueError("AMSDDataModule does not use the toolkit SpecAugment (specaugment.enabled=false)")
        if self.meta_label_mode is not None:
            raise ValueError("AMSDDataModule puts the sample index in the meta slot (meta_label_mode must be None)")
        if list(self.train_splits) == list(self.val_splits):
            raise ValueError("AMSDDataModule tells train/val apart by split name -- train_splits and val_splits cannot be equal")
        if not self.features_cfg.cache_dir:
            raise ValueError("AMSDDataModule requires features.cache_dir (a per-padding-setting subdirectory is appended)")
        self.amsd_freqm, self.amsd_timem = int(amsd_freqm), int(amsd_timem)
        self.amsd_mean, self.amsd_std = float(amsd_mean), float(amsd_std)
        self.amsd_return_index = bool(amsd_return_index)
        # Waveform preprocessing settings absent from the cache key (processor config) go into the directory name, so a changed padding assumption does not reuse the old cache
        self.pad_cache_subdir = f"pad-{self.pad_type}-{self.desired_length_sec}-fade{self.fade}"
        self.features_cfg.cache_dir = str(Path(self.features_cfg.cache_dir) / self.pad_cache_subdir)

    def _is_train_base(self, base_ds) -> bool:
        return _splits_of(base_ds) == [str(s) for s in self.train_splits]

    def _wrap_with_features(self, base_ds, *, feature_transform=None, meta_label_fn=None, split_role=None):
        inner = super()._wrap_with_features(base_ds, feature_transform=None, meta_label_fn=None, split_role=split_role)
        training = self._is_train_base(base_ds)  # decided explicitly by split name (independent of setup() call order)
        return AMSDFeatureDataset(inner, training=training, freqm=self.amsd_freqm, timem=self.amsd_timem,
                                  mean=self.amsd_mean, std=self.amsd_std, return_index=training and self.amsd_return_index)

    def _warmup_feature_cache(self, ds, *, split_name: str) -> None:  # warm the cache through the IndexFeatureDataset inside the wrapper
        inner = getattr(ds, "inner", ds)
        super()._warmup_feature_cache(inner, split_name=split_name)

    @property
    def num_train_samples(self) -> Optional[int]:
        return None if self.train_ds is None else len(self.train_ds)
