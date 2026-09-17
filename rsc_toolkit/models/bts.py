# Portions of this file are ported from BTS (https://github.com/kaen2891/bts), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Metadata-text fusion CLAP from the BTS paper (Interspeech 2024) — exact port of the official code.

Sources:
- Paper: BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound
  Classification (https://arxiv.org/abs/2406.06786)
- Official code: https://github.com/kaen2891/bts models/clap.py PretrainedCLAP + main.py set_model()
  (--model_type ClapModel --clap_final concat --meta_mode all)

Architecture: HF ClapModel (laion/clap-htsat-unfused; audio HTSAT + text RoBERTa, each with a 512-d
projection and L2 normalization) → text_embeds ⊕ audio_embeds (1024-d concat, or weighted sum for add)
→ nn.Linear(1024, n_cls). The text is a sentence built from patient metadata (age, sex, chest location, device)
(datasets/icbhi/meta_description.py). The original tokenizes it in the dataset __getitem__
(padding="max_length", max_length=64); here the model receives the batch meta vector and
encodes it the same way with the same tokenizer (numerically identical).

Kept from the original:
- The attribute name audio_features (original PretrainedCLAP.audio_features) is kept so the authors'
  checkpoints (best.pth: model=audio_features.*, classifier=weight/bias) load without key remapping.
- The text encoder is fine-tuned jointly during training (the original optimizer gets all of model.parameters()).
- The loss is unweighted CE (original set_model: nn.CrossEntropyLoss()).

Version difference: ClapModel in transformers 5.x only accepts 4D input_features (B, 1, T, F)
(the original passed (B, T, F) with an older version). A channel axis is added for 3D input.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from rsc_toolkit.datasets.icbhi.meta_description import (
    describe_meta7,
    generate_meta_description,
    meta_fields_from_meta7,
    select_meta_fields,
)

logger = logging.getLogger(__name__)


def _as_4d(input_features: torch.Tensor) -> torch.Tensor:
    return input_features.unsqueeze(1) if input_features.dim() == 3 else input_features


@ModelFactory.register_model_class("bts")
class BTSModel(BaseRSCModel, nn.Module):
    accepts_meta = ("bts_meta7",)  # batch[2] = meta7 float vector (bts_precomputed · btscard_sprs), also required for validation

    def __init__(
        self,
        n_cls: int = 4,
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feat_dim: int = 512,
        clap_final: str = "concat",     # "concat" | "add"
        te_alpha: float = 0.5,          # text weight for add (original default)
        meta_mode: str = "all",
        text_max_length: int = 64,      # original icbhi_dataset.py:189
        # Metadata sentence vocabulary. "icbhi" = original (7 chest locations + 4 stethoscopes).
        # "sprsound" = SPRSound version from the authors' util/sprsound_util.py (locations p1~p4, no stethoscope) —
        # use only when training directly on SPRSound. Without switching, nonexistent chest locations would be generated.
        meta_dialect: str = "icbhi",
    ):
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        if clap_final not in {"concat", "add"}:
            raise ValueError(f"clap_final must be concat|add, got {clap_final!r}")
        self.audio_features = ClapModel.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim
        self.clap_final = clap_final
        self.te_alpha = te_alpha
        self.meta_mode = meta_mode
        if meta_dialect not in {"icbhi", "sprsound"}:
            raise ValueError(f"meta_dialect must be icbhi|sprsound, got {meta_dialect!r}")
        self.meta_dialect = meta_dialect
        self.text_max_length = int(text_max_length)
        head_dim = final_feat_dim * 2 if clap_final == "concat" else final_feat_dim
        self.classifier = nn.Linear(head_dim, n_cls)  # original main.py:191
        # the original uses ClapProcessor(...).tokenizer — use the same object
        self.tokenizer = ClapProcessor.from_pretrained(pretrained_name).tokenizer

    # ----------------------------------------------------------------- text
    def meta_field_dicts(self, meta7_batch: torch.Tensor) -> list:
        """meta7 → sentence field dicts. Vocabulary and sentence template depend on the dialect."""
        if self.meta_dialect == "sprsound":
            from rsc_toolkit.datasets.sprsound_ood import meta_fields_from_meta7_sprsound

            return [meta_fields_from_meta7_sprsound(m.tolist()) for m in meta7_batch]
        return [select_meta_fields(meta_fields_from_meta7(m.tolist()), self.meta_mode) for m in meta7_batch]

    def render_meta_sentence(self, fields: dict) -> str:
        """Sentence generator. SPRSound uses the authors' util/sprsound_util.py version (no stethoscope term)."""
        if self.meta_dialect == "sprsound":
            from rsc_toolkit.datasets.sprsound_ood import generate_meta_description_sprsound

            return generate_meta_description_sprsound(
                age=fields["age"] if fields["age"] is not None else "unknown",
                sex=fields["sex"] if fields["sex"] is not None else "unknown",
                loc=fields["loc"] if fields["loc"] is not None else "unknown",
            )
        return generate_meta_description(**fields)

    def describe(self, meta7_batch: torch.Tensor) -> list:
        if self.meta_dialect == "sprsound":
            return [self.render_meta_sentence(d) for d in self.meta_field_dicts(meta7_batch)]
        return [describe_meta7(m.tolist(), self.meta_mode) for m in meta7_batch]

    def encode_meta(self, meta7_batch: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text=self.describe(meta7_batch), return_tensors="pt",
            padding="max_length", max_length=self.text_max_length,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ------------------------------------------------------------------ model
    def forward_features(self, input_features, input_ids, attention_mask) -> torch.Tensor:
        """Original PretrainedCLAP.forward (models/clap.py:26-36)."""
        out = self.audio_features(
            input_ids=input_ids, attention_mask=attention_mask, input_features=_as_4d(input_features)
        )
        text_embeds, audio_embeds = out.text_embeds, out.audio_embeds
        if self.clap_final == "concat":
            return torch.cat((text_embeds, audio_embeds), dim=-1)
        return (text_embeds * self.te_alpha) + (audio_embeds * (1 - self.te_alpha))

    def forward(self, x: torch.Tensor, meta7: torch.Tensor) -> torch.Tensor:
        ids, mask = self.encode_meta(meta7, x.device)
        return self.classifier(self.forward_features(x, ids, mask))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if meta_labels is None:
            raise ValueError("BTS requires meta (7 fields) in the batch — use data=bts_precomputed")
        logits = self.forward(audio_batch, meta_labels)
        return logits, loss_func(logits, labels)

    # ----------------------------------------------------- authors' checkpoint
    def load_reference_checkpoint(self, path) -> None:
        """
        Strictly loads the original save_model output (best.pth: {'model': audio_features.*, 'classifier': {...}})
        without key remapping (this also proves the architectures are identical).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model"])
        state.update({f"classifier.{k}": v for k, v in ckpt["classifier"].items()})
        self.load_state_dict(state, strict=True)
        logger.info("Loaded BTS authors' checkpoint: %s (epoch %s)", path, ckpt.get("epoch"))
