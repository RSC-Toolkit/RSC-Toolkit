import torch.nn as nn

from .base import BaseRSCModel
from .backbone import clap



class BTS(BaseRSCModel, nn.Module):
    # TODO can move this code to clap class and just import there.
    # In case of BTS, only ClapModel is used

    def __init__(
        self,
        n_cls: int,
        pretrained_name: str = "laion/clap-htsat-unfused",
        clap_backbone_model: str = "PretrainedCLAP",
        feature_fusion_mode: str = "concat",
        final_feature_dim: str = 512,
        ):

        self.clap_backbone_model = getattr(clap, clap_backbone_model)

        self.clap_backbone = self.clap_backbone_model(
            pretrained_name=pretrained_name,
            final_feature_dim=final_feature_dim,
        )
        
        if feature_fusion_mode == 'concat':
            self.classifier = nn.Linear(final_feature_dim * 2, n_cls)
        elif feature_fusion_mode == 'add':
            self.classifier = nn.Linear(final_feature_dim, n_cls)
        else:
            raise NotImplementedError
