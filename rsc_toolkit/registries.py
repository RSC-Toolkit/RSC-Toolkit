"""Registry hub for all components — the lazy declarations here are the single source of truth for built-in entries.

This module imports nothing heavy. Actual imports happen on first lookup, so if one paper module's
optional dependency (librosa, etc.) is missing, only that entry fails and the rest keep working.

Third parties register into the same registries with decorators:

    from rsc_toolkit.registries import MODELS, LOSSES
    @MODELS.register("my_model")
    class MyModel(BaseRSCModel, nn.Module): ...
    @LOSSES.register("focal")
    class FocalLoss(nn.Module): ...

Registering modules are loaded via the entry-point group `rsc_toolkit.plugins` or the `RSC_TOOLKIT_PLUGINS`
environment variable (see rsc_toolkit.registry).
"""
from __future__ import annotations

from rsc_toolkit.registry import Registry

# Models are registered by package scan (decorators in each module). base/factory are not models.
MODELS = Registry("model", scan_packages=("rsc_toolkit.models",), scan_exclude=("base", "factory"))
DATASETS = Registry("dataset")            # download, index, CLI (DatasetFactory delegates here)
DATAMODULES = Registry("datamodule")
LOSSES = Registry("loss")
METRICS = Registry("metric")
OPTIMIZERS = Registry("optimizer")
ADAPTERS = Registry("adapter")
PROCESSORS = Registry("processor")        # data.params.features.kind
AUGMENTATIONS = Registry("augmentation")  # feature-space transform
META_LABELS = Registry("meta_label")      # data.params.meta_label_mode

# ── datamodule ─────────────────────────────────────────────────────────────────────────
DATAMODULES.register_lazy("index", "rsc_toolkit.trainer.data.index_datamodule:IndexDataModule")
# training.pt/test.pt released by the authors (BTS, Audio-CLAP, BTS-CARD)
DATAMODULES.register_lazy("bts_precomputed", "rsc_toolkit.datasets.bts_precomputed:BTSPrecomputedDataModule")
# RespireNet 8 s chunking + magma images (+ optional CBA) (microsoft/RespireNet)
DATAMODULES.register_lazy("respirenet", "rsc_toolkit.datasets.respirenet_chunks:RespireNetDataModule")
# PC-MCL dynamic concatenation / PM-pair training set + IndexDataset evaluation (wa976/PC-MCL)
DATAMODULES.register_lazy("pcmcl", "rsc_toolkit.datasets.pcmcl_concat:PCMCLDataModule")
# AMS-D raw fbank cache + per-sample torchaudio masking/normalization + sample-index meta (CcnNnn/AMS-D)
DATAMODULES.register_lazy("amsd", "rsc_toolkit.datasets.amsd_data:AMSDDataModule")
# GaP-Aug class-balanced batch sampler (4 per class, 500-step epoch) — re-implemented from the paper description
DATAMODULES.register_lazy("gapaug", "rsc_toolkit.datasets.gapaug_data:GaPAugDataModule")
# Lungmix band-pass + SPRSound 7→4 derived labels (Tigerdwgth/ICASSP-2025-Lungmix, re-implemented from the paper description)
DATAMODULES.register_lazy("lungmix", "rsc_toolkit.datasets.lungmix_data:LungmixDataModule")
# Prototype (ICASSP 2022) 4 kHz band-pass, 4 s random/center crop + balanced class weight
DATAMODULES.register_lazy("protonet", "rsc_toolkit.datasets.protonet_data:ProtoNetDataModule")
# Domain Transfer (ICASSP 2022) 8 kHz, circular padding, splice, waveform augmentation
DATAMODULES.register_lazy("domaintransfer", "rsc_toolkit.datasets.domaintransfer_data:DomainTransferDataModule")
# QLung: AQS float as the third batch element (RSC-Toolkit/QLung 46408dd)
DATAMODULES.register_lazy("qlung", "rsc_toolkit.datasets.qlung_data:QLungDataModule")
# BTS-family SPR→SPR training: attaches the meta7-shaped vector to both train and val batches (the index path does not attach meta to val)
DATAMODULES.register_lazy("btscard_sprs", "rsc_toolkit.datasets.btscard_sprs:BTSCARDSprsDataModule")

# ── loss / metric / optimizer ─────────────────────────────────────────────────────────────
LOSSES.register_lazy("cross_entropy", "torch.nn:CrossEntropyLoss")
LOSSES.register_lazy("cross_entropy_from_counts", "rsc_toolkit.trainer.losses:WeightedCrossEntropyFromCounts")
LOSSES.register_lazy("cross_entropy_from_prior", "rsc_toolkit.trainer.losses:WeightedCrossEntropyFromPrior")
LOSSES.register_lazy("bce_with_logits", "torch.nn:BCEWithLogitsLoss")  # PC-MCL 3-label main loss (wa976/PC-MCL main.py:421)

METRICS.register_lazy("topk_accuracy", "rsc_toolkit.metrics.topk_accuracy:TopKAccuracy")
METRICS.register_lazy("icbhi_score", "rsc_toolkit.metrics.icbhi.score:ICBHIScore")
# AMS-D paper binary (non-standard) Se/Sp/Score (CcnNnn/AMS-D utils/metric.py)
METRICS.register_lazy("icbhi_paper_binary", "rsc_toolkit.metrics.icbhi.binary_score:PaperBinaryScore")

OPTIMIZERS.register_lazy("adam", "torch.optim:Adam")
OPTIMIZERS.register_lazy("adamw", "torch.optim:AdamW")
OPTIMIZERS.register_lazy("sgd", "torch.optim:SGD")

# ── adapter ────────────────────────────────────────────────────────────────────────────
ADAPTERS.register_lazy("pl", "rsc_toolkit.trainer.adapters.pl_adapter:PLAdapter")
# M2D fine-tuning quirk: split the batch into K parts and accumulate forward/backward (nttcslab/m2d ce.py)
ADAPTERS.register_lazy("split_iter", "rsc_toolkit.trainer.adapters.split_iter_adapter:SplitIterAdapter")

# ── processor — features.kind ──────────────────────────────────────────────────────────────
# Built via the looked-up class's from_options(opts, sampling_rate=, feature_cache=). get_processor_config()
# (= cache key) does not depend on the construction path, so existing caches still hit.
PROCESSORS.register_lazy("clap", "rsc_toolkit.processors.clap:CLAPProcessor")
PROCESSORS.register_lazy("mel", "rsc_toolkit.processors.mel:MelBankAudioProcessor")
PROCESSORS.register_lazy("fbank", "rsc_toolkit.processors.mel:MelBankAudioProcessor")  # alias (AST-family naming)
PROCESSORS.register_lazy("sclmeta_mel", "rsc_toolkit.processors.sclmeta_mel:SCLMetaMelProcessor")
PROCESSORS.register_lazy("waveform", "rsc_toolkit.processors.waveform:WaveformProcessor")  # no cache

# ── augmentation (feature-space transform) ──────────────────────────────────────────────
AUGMENTATIONS.register_lazy("specaugment", "rsc_toolkit.augmentations.specaugment:SpecAugment")

# ── meta label (data.params.meta_label_mode) ─────────────────────────────────────────────
# Registered objects are factories: called without arguments, they return an `item -> int` function.
META_LABELS.register_lazy("dev", "rsc_toolkit.datasets.meta_labels:build_dev")          # 4 ICBHI stethoscopes (sgscl family)
META_LABELS.register_lazy("patient", "rsc_toolkit.datasets.meta_labels:build_patient")  # patient ID (PAFA)
META_LABELS.register_lazy("sa", "rsc_toolkit.datasets.meta_labels:build_sa")            # sex × age group (SCL-metadata ICBHI)
META_LABELS.register_lazy("sa_sprs", "rsc_toolkit.datasets.meta_labels:build_sa_sprs")  # gender×age_group (SCL-metadata SPRS)

# ── dataset (raw datasets: download, index, CLI) ───────────────────────────────────────────
DATASETS.register_lazy("icbhi", "rsc_toolkit.datasets.icbhi.icbhi_dataset:ICBHIDataset")
DATASETS.register_lazy("sprsound", "rsc_toolkit.datasets.sprsound.sprsound_dataset:SPRSoundDataset")
DATASETS.register_lazy("hflungv1", "rsc_toolkit.datasets.hflungv1.hflungv1_dataset:HFLungV1Dataset")
DATASETS.register_lazy("toy", "rsc_toolkit.datasets.toy_dataset:ToyDataset")  # bundled fixture (sample_data)

ALL_REGISTRIES = {
    r.kind: r
    for r in (MODELS, DATASETS, DATAMODULES, LOSSES, METRICS, OPTIMIZERS, ADAPTERS, PROCESSORS, AUGMENTATIONS, META_LABELS)
}

__all__ = [
    "MODELS", "DATASETS", "DATAMODULES", "LOSSES", "METRICS", "OPTIMIZERS",
    "ADAPTERS", "PROCESSORS", "AUGMENTATIONS", "META_LABELS", "ALL_REGISTRIES",
]
