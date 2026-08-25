"""
Audio-CLAP — BTS 논문(kaen2891/bts)의 오디오 전용 기준선. 공식 코드 충실 이식 (P0).

원본 main.py set_model(--model_type ClapAudioModelWithProjection):
  model = PretrainedCLAPWithProjection('laion/clap-htsat-unfused', 512)  (models/clap.py:6-16)
  classifier = nn.Linear(512, n_cls)
forward 는 ClapAudioModelWithProjection(input_features).audio_embeds (512-d, L2 정규화).

transformers 5.x 는 input_features 를 (B, 1, T, F) 4D 로만 받는다 — 3D 면 채널 축을 붙인다.
"""
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from .backbone.clap import PretrainedCLAPWithProjection


@ModelFactory.register_model_class("audioclap")
class AudioCLAP(BaseRSCModel, nn.Module):
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
        # meta_labels 는 bts_precomputed datamodule 이 붙이지만 오디오 전용 기준선은 쓰지 않는다
        output = self.forward(audio_batch)
        return output, loss_func(output, labels)
