from transformers import ClapModel, ClapAudioModelWithProjection
import torch
import torch.nn as nn


class PretrainedCLAPWithProjection(nn.Module):
    def __init__(self, pretrained_name, final_feat_dim):
        super().__init__()

        self.pretrained = pretrained_name
        self.audio_features = ClapAudioModelWithProjection.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim
        
    def forward(self, x):
        if x.dim() == 3:  # (B, T, F) -> (B, 1, T, F): transformers 5.x accepts 4D input only
            x = x.unsqueeze(1)
        x = self.audio_features(x)
        return x.audio_embeds
    

# class PretrainedCLAP(nn.Module):
#     def __init__(self, pretrained_name, final_feat_dim):
#         super().__init__()

#         self.pretrained = pretrained_name
#         self.audio_features = ClapModel.from_pretrained(pretrained_name)
#         self.final_feat_dim = final_feat_dim
        
#     def forward(self, x, args=None, alpha=None, training=False):
#         text_inputs, attention_mask, audio_inputs = x
#         x = self.audio_features(input_ids=text_inputs, attention_mask=attention_mask, input_features=audio_inputs)
        
#         text_embeds = x.text_embeds
#         audio_embeds = x.audio_embeds
#         if args.clap_final == 'concat': # TODO why this?
#             return torch.cat((text_embeds, audio_embeds), dim=-1)
#         elif args.clap_final == 'add':
#             return (text_embeds * args.te_alpha) + (audio_embeds * (1 - args.te_alpha))
