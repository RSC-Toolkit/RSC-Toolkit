# Portions of this file are ported from BTS (https://github.com/kaen2891/bts), which does not include a license file.
# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Audio-CLAP — audio-only baseline of the BTS paper (kaen2891/bts). Exact port of the official code.

Original main.py set_model(--model_type ClapAudioModelWithProjection):
  model = PretrainedCLAPWithProjection('laion/clap-htsat-unfused', 512)  (models/clap.py:6-16)
  classifier = nn.Linear(512, n_cls)
forward is ClapAudioModelWithProjection(input_features).audio_embeds (512-d, not normalized —
it ends with audio_projection(pooler_output); L2 normalization exists only in the ClapModel.get_audio_features path).

transformers 5.x only accepts 4D input_features (B, 1, T, F) — a channel axis is added for 3D input.
"""
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .backbone.clap import PretrainedCLAPWithProjection


@ModelFactory.register_model_class("audioclap")
class AudioCLAP(BaseRSCModel, nn.Module):
    accepts_meta = ("bts_meta7",)  # bts_precomputed attaches meta7, but the audio-only baseline ignores it

    def __init__(
        self,
        n_cls: int,
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feature_dim: int = 512,
    ):
        super().__init__()
        self.backbone = PretrainedCLAPWithProjection(
            pretrained_name=pretrained_name,
            final_feat_dim=final_feature_dim,
        )
        self.classifier_head = nn.Linear(final_feature_dim, n_cls)

    def forward(self, x):
        return self.classifier_head(self.backbone(x))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        # meta_labels is attached by the bts_precomputed datamodule, but the audio-only baseline does not use it
        output = self.forward(audio_batch)
        return output, loss_func(output, labels)
