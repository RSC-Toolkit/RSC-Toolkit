import torch.nn as nn

from .base import BaseRSCModel
from .backbone.clap import PretrainedCLAPWithProjection



class AudioCLAP(BaseRSCModel, nn.Module): # shuold inherit from nn.Module
    # TODO can move this code to clap class and just import there.
    # In case of BTS, only ClapModel is used

    # https://github.com/RSC-Toolkit/BTS/blob/8ec14c7145795858d414b5430e8ae8c218f0e284/main.py#L244

    def __init__(
        self,
        n_cls: int,
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feature_dim: str = 512,
        ):
        super().__init__()
        self.backbone = PretrainedCLAPWithProjection(
            pretrained_name=pretrained_name,
            final_feat_dim=final_feature_dim,
        )
        self.classifier_head = nn.Linear(final_feature_dim, n_cls)
        
    def training_forward(self, audio_batch, labels, loss_func):
        features = self.backbone(audio_batch)
        output = self.classifier_head(features)
        return output, loss_func(output, labels)