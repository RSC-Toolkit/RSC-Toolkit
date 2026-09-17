"""Dataset <-> model batch contract, written down in one place.

This contract is otherwise implicit across four places: the IndexDataModule docstring (dict items), IndexFeatureDataset (positional tuple),
collate (4th element = meta) and PLAdapter (`batch[0].squeeze(1)`, `batch[2] = meta_labels`); a custom dataset would require
reverse-engineering all of them. This module only documents it **without changing anything** -- the batch layout itself is fixed
because 20+ ported models depend on it.

1. Index dataset item (`IndexDataset.__getitem__`): dict
       {"audio": Tensor(1, L) or (L,), "label": int | None, "sample_id": str, "meta": dict (optional), ...}
   Required index CSV columns: `rsc_toolkit.datasets.sample_index.REQUIRED_INDEX_COLUMNS`.
   `sample_level` vocabulary: `SAMPLE_LEVELS` (cycle | clip).

2. Feature wrapper item (`IndexFeatureDataset.__getitem__`): positional tuple
       (feats, label, sample_id[, meta])
   meta is appended as the 4th element only when the datamodule provides `meta_label_fn`.

3. Batch (`rsc_toolkit.datasets.collate`): positional tuple
       (x, y[, meta])
   - x = feats stack (B, ...) -- PLAdapter drops the channel axis with `batch[0].squeeze(1)`
   - y = long (B,)  (only PC-MCL training batches use float (B, 3) multi-label -- see pcmcl in META_KINDS below)
   - meta = 3rd element. **Its meaning differs per datamodule** -- so the datamodule declares `meta_kind`, the model declares
     `accepts_meta`, and PLAdapter.setup checks them against each other (rsc_toolkit.trainer.adapters.pl_adapter.check_meta_contract).

4. Model (`BaseRSCModel.training_forward(audio_batch, labels, loss_func, **kwargs)`): `(logits, loss)`.
   meta arrives as the `meta_labels=` keyword. Models that do not use it still accept `**kwargs`.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

SAMPLE_LEVELS: Tuple[str, ...] = ("cycle", "clip")

# Table of meta_kinds in the tree -- what each datamodule sends as batch[2]. A new datamodule declares its kind as a class attribute
# (registration here is documentation only), and receiving models list that name in accepts_meta.
META_KINDS: Dict[str, Dict[str, Any]] = {
    "dev": {"dtype": "long (B,)", "meaning": "ICBHI stethoscope id 0..3", "emitter": "IndexDataModule(meta_label_mode=dev)", "receivers": ("ast_sgscl", "resnet18/34_sft", "efficientnet_b0_sft", "cnn6_sft")},
    "patient": {"dtype": "long (B,)", "meaning": "patient number", "emitter": "IndexDataModule(meta_label_mode=patient)", "receivers": ("beats",)},
    "sa": {"dtype": "long (B,)", "meaning": "sex×age group 0..3 (ICBHI)", "emitter": "IndexDataModule(meta_label_mode=sa)", "receivers": ("cnn6_sclmeta",)},
    "sa_sprs": {"dtype": "long (B,)", "meaning": "gender×age_group 0..3 (SPRSound)", "emitter": "IndexDataModule(meta_label_mode=sa_sprs)", "receivers": ("cnn6_sclmeta",)},
    "qlung_aqs": {"dtype": "float64 (B,)", "meaning": "AQS acoustic quality score", "emitter": "QLungDataModule", "receivers": ("ast_qlung", "clap_qlung")},
    "pcmcl_ssl_label": {"dtype": "long (B,) ∈ {-1,0,1}", "meaning": "PC-MCL pair label (training y is float (B,3) multi-label)", "emitter": "PCMCLDataModule", "receivers": ("ast_pcmcl", "beats_pcmcl")},
    "bts_meta7": {"dtype": "float (B, 7)", "meaning": "BTS 7-field meta vector (both train and val)", "emitter": "BTSPrecomputedDataModule, BTSCARDSprsDataModule", "receivers": ("bts", "bts_card", "audioclap")},
    "amsd_sample_index": {"dtype": "long (B,)", "meaning": "training sample index (per-sample buffers)", "emitter": "AMSDDataModule", "receivers": ("ast_amsd",)},
    "respirenet_chunk_index": {"dtype": "long (B,)", "meaning": "chunk index (for the cycle-voting callback)", "emitter": "RespireNetDataModule", "receivers": ("respirenet",)},
    "lungmix_valid_len": {"dtype": "long (B,)", "meaning": "original segment length before padding (samples)", "emitter": "LungmixDataModule", "receivers": ("ast_lungmix",)},
}


class Batch(NamedTuple):
    """Named view of the batch tuple -- wrap with `Batch.from_tuple(batch)` to read it (the layout stays a plain tuple)."""

    x: torch.Tensor
    y: torch.Tensor
    meta: Optional[torch.Tensor] = None

    @classmethod
    def from_tuple(cls, batch) -> "Batch":
        if len(batch) < 2 or len(batch) > 3:
            raise ValueError(f"batch must have 2~3 elements (x, y[, meta]): len={len(batch)}")
        return cls(*batch)


def describe_batch(batch) -> Dict[str, Any]:
    """For debugging: shape and dtype of each element (including meta if present)."""
    b = Batch.from_tuple(batch)
    out = {"x": (tuple(b.x.shape), str(b.x.dtype)), "y": (tuple(b.y.shape), str(b.y.dtype))}
    if b.meta is not None:
        out["meta"] = (tuple(b.meta.shape), str(b.meta.dtype))
    return out


__all__ = ["SAMPLE_LEVELS", "META_KINDS", "Batch", "describe_batch"]
